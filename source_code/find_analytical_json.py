import argparse
import time
from pathlib import Path
import cv2, io
import torch
import torch.backends.cudnn as cudnn
import requests
import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time, json, math, pickle, os
from scipy.spatial import KDTree
from webcolors import (
    CSS3_HEX_TO_NAMES,
    hex_to_rgb,
)
from rembg.bg import remove
from skimage.color import rgb2lab, deltaE_cie76
from sklearn.cluster import KMeans
from PIL import Image, ImageFile
from collections import defaultdict, Counter
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

#from custom library
from find_dominant_color import get_colors
from skin_detector       import skinDetector

ImageFile.LOAD_TRUNCATED_IMAGES = True

@torch.no_grad()
def detect(opt):
    source, weights, view_img, save_txt, imgsz, max_items = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.max_items
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    save_img = False
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))
    s_namezz = str(source).split('/')[-1].split('.')[0]
    im = 0
    img_count = 0
    all_detected_names = []
    preds = []
    detected_images = []
    cloth_details = []
    unique_cloth_names = set()
    write_count = 0
    total_time_count = 0
    frames_count_num = 0
    analytics = {'analyticsInfo': []}
    gender_dicts = defaultdict(int)
    detected_obj_frame_count = defaultdict(int)
    detected_obj_time_count = defaultdict(int)
    detected_obj_max_conf = defaultdict(int)
    detected_obj_total_conf = defaultdict(int)
    detected_obj_max_conf_frame_num = defaultdict(int)
    detected_obj_max_conf_image = defaultdict(list)
    detected_obj_max_hsv = defaultdict(tuple)
    detected_obj_max_rgb = defaultdict(tuple)
    detected_obj_max_hsv_group = defaultdict(tuple)
    detected_obj_max_color = defaultdict(str)
    H_range = {'H1': (0, 10), 'H2': (10, 20), 'H3': (20, 30), 'H4': (30, 40), 'H5': (40, 50), 'H6': (50, 60),
               'H7': (60, 70), 'H8': (70, 80), 'H9': (80, 90), 'H10': (90, 100), 'H11': (100, 110), 'H12': (110, 120),
               'H13': (120, 130), 'H14': (130, 140), 'H15': (140, 150), 'H16': (150, 160), 'H17': (160, 170),
               'H18': (170, 180)}
    S_range = {'S1': (0, 39), 'S2': (39, 60), 'S3': (60, 88), 'S4': (88, 121), 'S5': (121, 157), 'S6': (157, 180),
               'S7': (180, 213), 'S8': (213, 256)}
    V_range = {'V1': (0, 42), 'V2': (42, 57), 'V3': (57, 75), 'V4': (75, 101), 'V5': (101, 131), 'V6': (131, 167),
               'V7': (167, 206), 'V8': (206, 256)}
    female_clothes = ['full_cami_tops', 'full_tube_tops', 'regular_sleeveless_tops', 'half_tank_tops', 'half_cami_tops',
                      'floor_length_skirt', 'knee_length_skirt', 'half_dress', 'maxi_dress', 'tunic_tops',
                      'half_tube_tops','sleeved_crop_tops', 'mini_skirt', 'kurta', 'blouse', 'lehenga', 'saree']
    skin_exposing_clothes = ['normal_shorts', 'jeans_short', 'blouse', 'mini_skirt', 'sleeved_crop_tops',
                             'full_tank_tops', 'half_tube_tops', 'full_shirt',
                             'jeans_short', 'half_dress', 'full_tube_tops', 'full_cami_tops', 'half_shirt',
                             'half_tshirt', 'half_cami_tops', 'half_tank_tops', 'regular_sleeveless_tops']

    # Directories
    # save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    def displayCategory(features, category):
        s = ""
        if isinstance(features, dict) and isinstance(category, str):
            for k, v in features.items():
                if features[k] is not None:
                    s += features[k] + ' '
            s += category
            return str(s).strip()
        else:
            return "None"

    videoIndex = 1
    for img_ind, (path, img, im0s, vid_cap) in enumerate(dataset):
        frames_count_num += 1

        if img_ind % 100 != 0:
            continue

        im0s = cv2.cvtColor(im0s, cv2.COLOR_RGB2BGR)
        fps = vid_cap.get(cv2.CAP_PROP_FPS)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32

        pil_im = Image.fromarray(im0s)
        byt = io.BytesIO()
        pil_im.save(byt, 'PNG')
        f_value = byt.getvalue()
        result_im = remove(f_value)
        # imgs = Image.open(io.BytesIO(result_im)).convert("RGBA")
        imgs = Image.open(io.BytesIO(result_im))
        imgs.load()  # required for png.split()
        background = Image.new("RGB", imgs.size, (255, 255, 255))
        background.paste(imgs, mask=imgs.split()[3])  # 3 is the alpha channelim0s = np.asarray(background)
        im0s = np.asarray(background)
        im0s = cv2.cvtColor(im0s, cv2.COLOR_BGR2RGB)

        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms,
                                   max_det=opt.max_det)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Dictionaries for clothing and gender classification task
        m_count = 0
        f_count = 0
        cloth_center_dict = {}  # for storing center of the detected cloth
        male_range = {}  # for storing the x-axis range of male
        female_range = {}  # for storing the x-axis range of female
        cloth_gender = {}  # for storing the cloth and to which gender it belongs
        f_d = defaultdict(list)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            # save_path = str(save_dir / p.name)  # img.jpg
            # txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if opt.save_crop else im0  # for opt.save_crop
            im_new = im0.copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    # c is class index
                    # n is total objects detected of single class
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for i, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    al = [x.item() for x in xyxy]  # gets list of bboxes
                    x1, y1, x2, y2 = int(al[0]), int(al[1]), int(al[2]), int(al[3])
                    cent_cloth = (int(x1) + (int(x2) - int(x1)) // 2, int(y1) + (int(y2) - int(y1)) // 2)
                    tl, tr = (int(x1), int(y1)), (int(x2), int(y1))
                    box_wt = int(x2) - int(x1)
                    padding = int(0.1 * box_wt)
                    tl, tr = (int(x1) + padding, int(y1)), (int(x2) - padding, int(y1))

                    if names[int(cls)] == 'male' and conf.item() > 0.55:
                        m_count += 1
                        male_range[i] = (tl[0], tr[0])
                    if names[int(cls)] == 'female' and conf.item() > 0.55:
                        f_count += 1
                        female_range[i] = (tl[0], tr[0])

                    if not names[int(cls)] == 'male' and not names[int(cls)] == 'female':
                        cloth_center_dict[i] = cent_cloth

                if m_count > gender_dicts['M']:
                    gender_dicts['M'] = m_count
                if f_count > gender_dicts['F']:
                    gender_dicts['F'] = f_count

                for ind, cent in cloth_center_dict.items():
                    for m_gen, xm_range in male_range.items():
                        if cent[0] in range(xm_range[0], xm_range[1]):
                            # print(f'{item} is of {m_gen}')
                            f_d[ind].append('male')
                            break

                    for f_gen, xf_range in female_range.items():
                        if cent[0] in range(xf_range[0], xf_range[1]):
                            # print(f'{item} is of {f_gen}')
                            f_d[ind].append('female')
                            break

                for ind, c in cloth_center_dict.items():
                    try:
                        if len(f_d[ind]) > 1:
                            cloth_gender[ind] = 'U'
                        elif f_d[ind] == ['male']:
                            cloth_gender[ind] = 'M'
                        elif f_d[ind] == ['female']:
                            cloth_gender[ind] = 'F'
                        else:
                            cloth_gender[ind] = 'U'
                    except:
                        cloth_gender[ind] = 'U'

                # Write results
                for inde, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    try:
                        if names[int(cls)] == 'male' or names[int(cls)] == 'female':
                            continue
                        confidence_score = conf.item()
                        if (names[int(cls)] == 'shoes') or (names[int(cls)] == 'slippers') or (names[int(cls)] == 'heel'):
                            if confidence_score > 0.8:
                                al = [x.item() for x in xyxy]  # gets list of bboxes
                                hsv_tag = []
                                x1, y1, x2, y2 = int(al[0]), int(al[1]), int(al[2]), int(al[3])
                                detected_image = im_new[y1:y2, x1:x2]
                                detector = skinDetector(detected_image)
                                remskin_img = detector.find_skin()
                                white_pixels = np.logical_and(255 == remskin_img[:, :, 0],
                                                              np.logical_and(255 == remskin_img[:, :, 1],
                                                                             255 == remskin_img[:, :, 2]))
                                if np.sum(white_pixels) > (87 / 100 * remskin_img.shape[0] * remskin_img.shape[1]):
                                    continue
                                dominant_color = get_colors(remskin_img)  # returns BGR
                                dominant_color = dominant_color[::-1]
                                if dominant_color == (255, 255, 255):
                                    continue
                                bgr_equi = np.array([dominant_color[2], dominant_color[1], dominant_color[0]],
                                                    dtype='uint8').reshape(1, 1, 3)
                                h, s, v = cv2.cvtColor(bgr_equi, cv2.COLOR_BGR2HSV).squeeze()
                                for h_name, h_range in H_range.items():
                                    if h in range(h_range[0], h_range[1]):
                                        hsv_tag.append(h_name)
                                        break
                                for s_name, s_range in S_range.items():
                                    if s in range(s_range[0], s_range[1]):
                                        hsv_tag.append(s_name)
                                        break
                                for v_name, v_range in V_range.items():
                                    if v in range(v_range[0], v_range[1]):
                                        hsv_tag.append(v_name)
                                        break
                                hsv_g = "".join(hsv_tag)
                                det_name = str(cloth_gender[inde]) + ' ' + hsv_g + ' ' + names[int(cls)]
                                detected_obj_frame_count[det_name] += 1
                                detected_obj_time_count[det_name] += 1 / fps
                                detected_obj_total_conf[det_name] += confidence_score
                                total_time_count += 1 / fps

                                if confidence_score > detected_obj_max_conf[det_name]:
                                    detected_obj_max_conf[det_name] = confidence_score
                                    detected_obj_max_conf_frame_num[det_name] = frames_count_num
                                    detected_obj_max_conf_image[det_name] = detected_image
                                    detected_obj_max_hsv[det_name] = (h, s, v)
                                    detected_obj_max_rgb[det_name] = dominant_color
                                    detected_obj_max_hsv_group[det_name] = tuple(hsv_tag)
                                    detected_obj_max_color[det_name] = convert_rgb_to_names(dominant_color)

                                if (det_name not in preds):
                                    preds.append(det_name)

                        elif (names[int(cls)] == 'saree'):
                            if confidence_score > 0.88:
                                al = [x.item() for x in xyxy]  # gets list of bboxes
                                x1, y1, x2, y2 = int(al[0]), int(al[1]), int(al[2]), int(al[3])
                                hsv_tag = []
                                # selecting the middle region
                                detected_image = im_new[y1:y2, x1:x2]
                                temp_image = im_new[y1 + int(int(y2 - y1) * 0.22):y2, x1:x2]
                                dominant_color = get_colors(temp_image)  # returns BGR
                                dominant_color = dominant_color[::-1]
                                if dominant_color == (255, 255, 255):
                                    continue
                                bgr_equi = np.array([dominant_color[2], dominant_color[1], dominant_color[0]],
                                                    dtype='uint8').reshape(1, 1, 3)
                                h, s, v = cv2.cvtColor(bgr_equi, cv2.COLOR_BGR2HSV).squeeze()
                                for h_name, h_range in H_range.items():
                                    if h in range(h_range[0], h_range[1]):
                                        hsv_tag.append(h_name)
                                        break
                                for s_name, s_range in S_range.items():
                                    if s in range(s_range[0], s_range[1]):
                                        hsv_tag.append(s_name)
                                        break
                                for v_name, v_range in V_range.items():
                                    if v in range(v_range[0], v_range[1]):
                                        hsv_tag.append(v_name)
                                        break
                                hsv_g = "".join(hsv_tag)
                                det_name = 'F' + ' ' + hsv_g + ' ' + names[int(cls)]
                                detected_obj_frame_count[det_name] += 1
                                detected_obj_time_count[det_name] += 1 / fps
                                detected_obj_total_conf[det_name] += confidence_score
                                total_time_count += 1 / fps

                                if confidence_score > detected_obj_max_conf[det_name]:
                                    detected_obj_max_conf[det_name] = confidence_score
                                    detected_obj_max_conf_frame_num[det_name] = frames_count_num
                                    detected_obj_max_conf_image[det_name] = detected_image
                                    detected_obj_max_hsv[det_name] = (h, s, v)
                                    detected_obj_max_rgb[det_name] = dominant_color
                                    detected_obj_max_hsv_group[det_name] = tuple(hsv_tag)
                                    detected_obj_max_color[det_name] = convert_rgb_to_names(dominant_color)

                                if (det_name not in preds):
                                    preds.append(det_name)

                        else:
                            if confidence_score > 0.8:
                                al = [x.item() for x in xyxy]  # gets list of bboxes
                                x1, y1, x2, y2 = int(al[0]), int(al[1]), int(al[2]), int(al[3])
                                # selecting the middle region
                                if names[int(cls)] in ['jeans_pant', 'track_pant', 'baggy_pant', 'formal_pant']:
                                    hsv_tag = []
                                    detected_image = im_new[y1:y2, x1:x2]
                                    temp_image = im_new[y1 + int(int(y2 - y1) * 0.22):y2, x1:x2]
                                    dominant_color = get_colors(temp_image)  # returns BGR
                                    dominant_color = dominant_color[::-1]
                                    if dominant_color == (255, 255, 255):
                                        continue
                                    bgr_equi = np.array([dominant_color[2], dominant_color[1], dominant_color[0]],
                                                        dtype='uint8').reshape(1, 1, 3)
                                    h, s, v = cv2.cvtColor(bgr_equi, cv2.COLOR_BGR2HSV).squeeze()
                                    for h_name, h_range in H_range.items():
                                        if h in range(h_range[0], h_range[1]):
                                            hsv_tag.append(h_name)
                                            break
                                    for s_name, s_range in S_range.items():
                                        if s in range(s_range[0], s_range[1]):
                                            hsv_tag.append(s_name)
                                            break
                                    for v_name, v_range in V_range.items():
                                        if v in range(v_range[0], v_range[1]):
                                            hsv_tag.append(v_name)
                                            break
                                    hsv_g = "".join(hsv_tag)
                                    det_name = str(cloth_gender[inde]) + ' ' + hsv_g + ' ' + names[int(cls)]
                                    detected_obj_frame_count[det_name] += 1
                                    detected_obj_time_count[det_name] += 1 / fps
                                    detected_obj_total_conf[det_name] += confidence_score
                                    total_time_count += 1 / fps

                                    if confidence_score > detected_obj_max_conf[det_name]:
                                        detected_obj_max_conf[det_name] = confidence_score
                                        detected_obj_max_conf_frame_num[det_name] = frames_count_num
                                        detected_obj_max_conf_image[det_name] = detected_image
                                        detected_obj_max_hsv[det_name] = (h, s, v)
                                        detected_obj_max_rgb[det_name] = dominant_color
                                        detected_obj_max_hsv_group[det_name] = tuple(hsv_tag)
                                        detected_obj_max_color[det_name] = convert_rgb_to_names(dominant_color)

                                    if (det_name not in preds):
                                        preds.append(det_name)

                                elif names[int(cls)] in skin_exposing_clothes:
                                    hsv_tag = []
                                    detected_image = im_new[y1:y2, x1:x2]
                                    temp_image = im_new[y1 + int(int(y2 - y1) * 0.24):y2, x1:x2]
                                    detector = skinDetector(temp_image)
                                    remskin_img = detector.find_skin()
                                    white_pixels = np.logical_and(255 == remskin_img[:, :, 0],
                                                                  np.logical_and(255 == remskin_img[:, :, 1],
                                                                                 255 == remskin_img[:, :, 2]))
                                    if np.sum(white_pixels) > (87 / 100 * remskin_img.shape[0] * remskin_img.shape[1]):
                                        continue
                                    dominant_color = get_colors(remskin_img)  # returns BGR
                                    dominant_color = dominant_color[::-1]
                                    if dominant_color == (255, 255, 255):
                                        continue
                                    bgr_equi = np.array([dominant_color[2], dominant_color[1], dominant_color[0]],
                                                        dtype='uint8').reshape(1, 1, 3)
                                    h, s, v = cv2.cvtColor(bgr_equi, cv2.COLOR_BGR2HSV).squeeze()
                                    for h_name, h_range in H_range.items():
                                        if h in range(h_range[0], h_range[1]):
                                            hsv_tag.append(h_name)
                                            break
                                    for s_name, s_range in S_range.items():
                                        if s in range(s_range[0], s_range[1]):
                                            hsv_tag.append(s_name)
                                            break
                                    for v_name, v_range in V_range.items():
                                        if v in range(v_range[0], v_range[1]):
                                            hsv_tag.append(v_name)
                                            break
                                    hsv_g = "".join(hsv_tag)
                                    if names[int(cls)] in female_clothes:
                                        cloth_gender[inde] = 'F'
                                    det_name = cloth_gender[inde] + ' ' + hsv_g + ' ' + names[int(cls)]
                                    detected_obj_frame_count[det_name] += 1
                                    detected_obj_time_count[det_name] += 1 / fps
                                    detected_obj_total_conf[det_name] += confidence_score
                                    total_time_count += 1 / fps

                                    if confidence_score > detected_obj_max_conf[det_name]:
                                        detected_obj_max_conf[det_name] = confidence_score
                                        detected_obj_max_conf_frame_num[det_name] = frames_count_num
                                        detected_obj_max_conf_image[det_name] = detected_image
                                        detected_obj_max_hsv[det_name] = (h, s, v)
                                        detected_obj_max_rgb[det_name] = dominant_color
                                        detected_obj_max_hsv_group[det_name] = tuple(hsv_tag)
                                        detected_obj_max_color[det_name] = convert_rgb_to_names(dominant_color)

                                    if (det_name not in preds):
                                        preds.append(det_name)


                                else:
                                    hsv_tag = []
                                    detected_image = im_new[y1:y2, x1:x2]
                                    temp_image = im_new[y1 + int(int(y2 - y1) * 0.24):y2, x1:x2]
                                    dominant_color = get_colors(temp_image)  # returns BGR
                                    dominant_color = dominant_color[::-1]
                                    if dominant_color == (255, 255, 255):
                                        continue
                                    bgr_equi = np.array([dominant_color[2], dominant_color[1], dominant_color[0]],
                                                        dtype='uint8').reshape(1, 1, 3)
                                    h, s, v = cv2.cvtColor(bgr_equi, cv2.COLOR_BGR2HSV).squeeze()
                                    for h_name, h_range in H_range.items():
                                        if h in range(h_range[0], h_range[1]):
                                            hsv_tag.append(h_name)
                                            break
                                    for s_name, s_range in S_range.items():
                                        if s in range(s_range[0], s_range[1]):
                                            hsv_tag.append(s_name)
                                            break
                                    for v_name, v_range in V_range.items():
                                        if v in range(v_range[0], v_range[1]):
                                            hsv_tag.append(v_name)
                                            break
                                    hsv_g = "".join(hsv_tag)
                                    if names[int(cls)] in female_clothes:
                                        cloth_gender[inde] = 'F'
                                    det_name = cloth_gender[inde] + ' ' + hsv_g + ' ' + names[int(cls)]
                                    detected_obj_frame_count[det_name] += 1
                                    detected_obj_time_count[det_name] += 1 / fps
                                    detected_obj_total_conf[det_name] += confidence_score
                                    total_time_count += 1 / fps

                                    if confidence_score > detected_obj_max_conf[det_name]:
                                        detected_obj_max_conf[det_name] = confidence_score
                                        detected_obj_max_conf_frame_num[det_name] = frames_count_num
                                        detected_obj_max_conf_image[det_name] = detected_image
                                        detected_obj_max_hsv[det_name] = (h, s, v)
                                        detected_obj_max_rgb[det_name] = dominant_color
                                        detected_obj_max_hsv_group[det_name] = tuple(hsv_tag)
                                        detected_obj_max_color[det_name] = convert_rgb_to_names(dominant_color)

                                    if (det_name not in preds):
                                        preds.append(det_name)

                    except:
                        continue

            print(f'{s}Done. ({t2 - t1:.3f}s)')

    print(f'Done. ({time.time() - t0:.3f}s)')

    if len(preds) > 0:
        for pr in preds:
            gender_n = str(pr).split(' ')[0]
            cf = detected_obj_total_conf[pr] / detected_obj_frame_count[pr]
            nt = detected_obj_time_count[pr] / total_time_count
            frs = 0.2 * cf + 0.8 * nt
            aInfo = {'category': str(pr).split(' ')[2],
                     'features': {'color': detected_obj_max_color[pr],
                                  'h': int(detected_obj_max_hsv[pr][0]),
                                  's': int(detected_obj_max_hsv[pr][1]),
                                  'v': int(detected_obj_max_hsv[pr][2]),
                                  'hGrp': str(detected_obj_max_hsv_group[pr][0]),
                                  'sGrp': str(detected_obj_max_hsv_group[pr][1]),
                                  'vGrp': str(detected_obj_max_hsv_group[pr][2]),
                                  'r': int(detected_obj_max_rgb[pr][0]),
                                  'g': int(detected_obj_max_rgb[pr][1]),
                                  'b': int(detected_obj_max_rgb[pr][2]),
                                  'gender': gender_n, 'objectCount': detected_obj_frame_count[pr],
                                  'timeSec': detected_obj_time_count[pr], 'normalizedTime': nt,
                                  'averageConfidenceScore': cf, 'analyticsRankScore': frs,
                                  'maxConf': detected_obj_max_conf[pr],
                                  'maxConfFrameNumber': detected_obj_max_conf_frame_num[pr]}}
            analytics['analyticsInfo'].append(aInfo)
        # analytics['videoIndex'] = videoIndex
        # analytics['videoSource'] = str(source).strip()
        analytics_ = sorted(analytics['analyticsInfo'], key=lambda x: x['features']['analyticsRankScore'], reverse=True)
        dup_gen_cat = []
        dup_hsv = defaultdict(list)
        dup_gen_ind = defaultdict(list)

        hsv_c = [
            (int(item['features']['hGrp'][1:]), int(item['features']['sGrp'][1:]), int(item['features']['vGrp'][1:]))
            for item in analytics_]
        gen_cat = [str(item['features']['gender'] + item['category']) for item in analytics_]

        for k, v in dict(Counter(gen_cat)).items():
            if v > 1:
                dup_gen_cat.append(k)

        for e in dup_gen_cat:
            for i, c in enumerate(gen_cat):
                if c == e:
                    dup_gen_ind[e].append(i)

        for k, v in dup_gen_ind.items():
            for ind in v:
                dup_hsv[k].append(hsv_c[ind])

        invalid_indx = []
        try:
            for dup in dup_gen_cat:
                for i, val in enumerate(dup_hsv[dup]):
                    h_r = (val[0] - 2, val[0] + 3)
                    s_r = (val[1] - 2, val[1] + 3)
                    v_r = (val[2] - 2, val[2] + 3)
                    if i < len(val) - 1:
                        for r_v in dup_hsv[dup][i + 1:]:
                            if val[1] == 1 and r_v[1] == 1:  # For S
                                # print(dup, dup_hsv[dup])
                                # print(dup, dup_gen_ind[dup])
                                invalid_indx.append(dup_gen_ind[dup][dup_hsv[dup].index(r_v)])
                                dup_gen_ind[dup].remove(dup_gen_ind[dup][dup_hsv[dup].index(r_v)])
                                dup_hsv[dup].remove(r_v)
                                # print(dup, dup_hsv[dup])
                                # print(dup, dup_gen_ind[dup])
                            elif (val[2] in range(1, 3)) and (r_v[2] in range(1, 3)):  # For V
                                invalid_indx.append(dup_gen_ind[dup][dup_hsv[dup].index(r_v)])
                                dup_gen_ind[dup].remove(dup_gen_ind[dup][dup_hsv[dup].index(r_v)])
                                dup_hsv[dup].remove(r_v)
                            elif (r_v[0] in range(h_r[0], h_r[1])) and (r_v[1] in range(s_r[0], s_r[1])) and (
                                    r_v[2] in range(v_r[0], v_r[1])):
                                invalid_indx.append(dup_gen_ind[dup][dup_hsv[dup].index(r_v)])
                                dup_gen_ind[dup].remove(dup_gen_ind[dup][dup_hsv[dup].index(r_v)])
                                dup_hsv[dup].remove(r_v)

        except:
            pass
        print(s_namezz)
        new_analytics_ = []
        for i, val in enumerate(analytics_):
            if i in invalid_indx:
                continue
            new_analytics_.append(val)

        vnew_analytics_ = []
        for val in new_analytics_:
            if val['features']['normalizedTime'] < 0.03:
                continue
            vnew_analytics_.append(val)

        all_feat = []
        all_val = []
        dup_feat = []
        all_feat = [ele['features']['gender'] + ele['category'] for ele in vnew_analytics_]
        for ele in vnew_analytics_:
            all_val.append(
                ele['features']['gender'] + ele['category'] + '+' + ele['features']['hGrp'] + ele['features']['sGrp'] \
                + ele['features']['vGrp'])

        for k, v in dict(Counter(all_feat)).items():
            if v > gender_dicts[k[0]]:
                dup_feat.append(k)

        uniq_ind = set()
        uniq_feat = []
        if len(dup_feat) > 0:
            try:
                for feat in dup_feat:
                    for i, vl in enumerate(all_val):
                        if vl.split('+')[0][0] == 'U':
                            continue
                            # if (dict(Counter(uniq_feat)).get(vl.split('+')[0],0) <= gender_dicts[vl.split('+')[0][0]]):
                            #     uniq_ind.add(i)
                            #     uniq_feat.append(vl.split('+')[0])
                        elif (feat == vl.split('+')[0]) and (
                                dict(Counter(uniq_feat)).get(vl.split('+')[0], 0) < gender_dicts[vl.split('+')[0][0]]):
                            uniq_ind.add(i)
                            uniq_feat.append(vl.split('+')[0])
                        elif (feat != vl.split('+')[0]) and (
                                dict(Counter(uniq_feat)).get(vl.split('+')[0], 0) < gender_dicts[vl.split('+')[0][0]]):
                            uniq_ind.add(i)
                            uniq_feat.append(vl.split('+')[0])
            except:
                pass

        uniq_list = sorted(list(uniq_ind))
        if len(uniq_list) == 0:
            uniq_list = [i for i, v in enumerate(vnew_analytics_)]

        fnew_analytics_ = []
        for i, val in enumerate(vnew_analytics_):
            if i in uniq_list:
                fnew_analytics_.append(val)

        rm = []
        for ele in fnew_analytics_:
            if (ele['features']['h'] == 0) and (ele['features']['s'] == 0):
                rm.append(ele)
        if len(rm) > 0:
            for ele in rm:
                fnew_analytics_.remove(ele)
                
#         rmg = []
#         for ele in fnew_analytics_:
#             if (ele['features']['gender'] == 'U'):
#                 rmg.append(ele)
#         if len(rmg) > 0:
#             for ele in rmg:
#                 fnew_analytics_.remove(ele)

   
        file_name = source.split('/')[-1].replace('.mp4', '')
        print(file_name)

        try:
            
            if len(fnew_analytics_) == 0:
                
                
                analytics = {'analyticsInfo': []}
                with open(os.path.join('../analytics_json',f'{file_name}.json'), 'w') as f:
                    f.write(json.dumps(analytics))
                return analytics

            else:
                final_analytics = defaultdict(list)
                iii = 0
                for i, ele in enumerate(fnew_analytics_[:4]):
                    det_obj = ele['features']['gender'] + ' ' + ele['features']['hGrp'] + ele['features']['sGrp'] \
                              + ele['features']['vGrp'] + ' ' + ele['category']
                    rnk = i + 1
                    det_obj1 = ele['features']['gender'] + '_' + ele['features']['hGrp'] + ele['features']['sGrp'] \
                               + ele['features']['vGrp'] + '_' + ele['category']
                    f_name_image = f"{s_namezz}_analyticsRank_{rnk}_{det_obj1}.png"
                    ele['analyticsRank'] = rnk
                    ele['objectImage'] = f_name_image
                    try:
                        cv2.imwrite(os.path.join('../detected_image',f'{f_name_image}'), detected_obj_max_conf_image[det_obj])
                        final_analytics['analyticsInfo'].append(ele)
                    except:
                        iii += 1
                        if iii >= len(fnew_analytics_):
                            analytics = {'analyticsInfo': []}
                            with open(os.path.join('../analytics_json',f'{file_name}.json'), 'w') as f:
                                f.write(json.dumps(analytics))
                                return analytics
                        continue
                final_analytics = dict(final_analytics)
                with open(os.path.join('../analytics_json',f'{file_name}.json'), 'w') as f:
                    f.write(json.dumps(final_analytics))
                return final_analytics
        except:
            analytics = {'analyticsInfo': []}
            with open(os.path.join('../analytics_json',f'{file_name}.json'), 'w') as f:
                f.write(json.dumps(analytics))
                return analytics
    else:
        analytics = {'analyticsInfo': []}
        with open(os.path.join('../analytics_json',f'{file_name}.json'), 'w') as f:
            f.write(json.dumps(analytics))
            return analytics


def VideoAnalytics(video_path):
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', nargs='+', type=str,
                            default='../weights/best_v12.pt',
                            help='model.pt path(s)')
        parser.add_argument('--source', type=str, default=video_path, help='source')  # file/folder, 0 for webcam
        parser.add_argument('--img-size', type=int, default=256, help='inference size (pixels)')
        parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
        parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
        parser.add_argument('--max-det', type=int, default=1000, help='maximum number of detections per image')
        parser.add_argument('--max-items', type=int, default=200, help='maximum number of clothes to be detected')
        parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
        parser.add_argument('--view-img', action='store_true', help='display results')
        parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
        parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
        parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
        parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
        parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
        parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
        parser.add_argument('--augment', action='store_true', help='augmented inference')
        parser.add_argument('--update', action='store_true', help='update all models')
        parser.add_argument('--project', default='runs/detect', help='save results to project/name')
        parser.add_argument('--name', default='exp', help='save results to project/name')
        parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
        parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
        parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
        parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
        opt = parser.parse_args(args=[])
        check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                json_path = detect(opt=opt)
                strip_optimizer(opt.weights)
                return json_path
        else:
            json_path = detect(opt=opt)
            return json_path

    except Exception as e:
        print(e)
        return None

# VideoAnalytics('../test_videos/test32.mp4')