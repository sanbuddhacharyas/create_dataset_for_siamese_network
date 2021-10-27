from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import time
import trace
import sys
import pandas as pd
from tqdm import tqdm
from joblib import load, dump

def merge_related_group(rgb_colors, hsv_colors, rgb_hsv_color_per, number_of_colors):
    
    h_threshold = 10
    s_threshold = 30
    v_threshold = 35
    #find if the colors are of same group
    for x_ind in range(number_of_colors):
        for y_ind in range(number_of_colors):
            if (x_ind < y_ind) and (y_ind < len(hsv_colors)):
                if (abs(hsv_colors[x_ind][0].astype(np.int8) - hsv_colors[y_ind][0].astype(np.int8)) < h_threshold) and \
                (abs(hsv_colors[x_ind][1].astype(np.int8) - hsv_colors[y_ind][1].astype(np.int8)) < s_threshold) and \
                (abs(hsv_colors[x_ind][2].astype(np.int8) - hsv_colors[y_ind][2].astype(np.int8)) < v_threshold):

                    hsv_colors[x_ind]  = hsv_colors[x_ind] if (np.sum(hsv_colors[x_ind]) > np.sum(hsv_colors[y_ind])) else hsv_colors[y_ind]
                    hsv_colors.pop(y_ind)
                    
                    rgb_colors[x_ind]  = rgb_colors[x_ind] if (np.sum(rgb_colors[x_ind]) > np.sum(rgb_colors[y_ind])) else rgb_colors[y_ind]
                    rgb_colors.pop(y_ind)
                    
                    rgb_hsv_color_per[x_ind] = rgb_hsv_color_per[x_ind] + rgb_hsv_color_per[y_ind]
                    rgb_hsv_color_per.pop(y_ind)
                    
    return rgb_colors, hsv_colors, rgb_hsv_color_per

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def crop_images(img, width_ratio=0.4, u_height_ratio=0.2, l_heigh_ratio=0.2):
    
    img_ht, img_wt, layers = img.shape
    x , y = int(img_wt / 2), int(img_ht/ 2)

    x1 = int(x - width_ratio * img_wt)
    x2 = int(x + width_ratio * img_wt)
    y1 = int(y - u_height_ratio * img_ht)
    y2 = int(y + l_heigh_ratio * img_ht)
    
    return img[y1:y2, x1:x2]


def get_colors(image, number_of_colors=4, show_chart=False):
   
        
    try:
        modified_image = cv2.resize(image, (100, 100), interpolation = cv2.INTER_AREA)
        modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
        modified_image = np.where(modified_image <= 252, modified_image, 255)

        modified_image = modified_image[(modified_image!=np.array([[255, 255, 255]])).all(axis=1)]

        if list(modified_image)==[]:
            return [(255, 255, 255)]*3

        clf    = KMeans(n_clusters = number_of_colors)
        labels = clf.fit_predict(modified_image) 

        # sort to ensure correct color percentage
        counts = dict(sorted(Counter(labels).items()))

        center_colors = clf.cluster_centers_

        # We get ordered colors by iterating through the keys
        ordered_colors = [center_colors[i]  for i in counts.keys()]
        rgb_colors     = [ordered_colors[i] for i in counts.keys()]
        hsv_colors     = (cv2.cvtColor(np.stack(rgb_colors)[np.newaxis,...].astype(np.uint8), cv2.COLOR_RGB2HSV).squeeze())

        list_count_colors     = list(counts.values())
        total_color           = np.sum(list_count_colors)
        rgb_hsv_color_per     = [(color_counts / total_color) * 100 for color_counts in list_count_colors]

        
        rgb_colors, hsv_colors, rgb_hsv_color_per = merge_related_group(rgb_colors, list(hsv_colors), rgb_hsv_color_per, number_of_colors)

        rgb_hsv_sorted_ind    = np.argsort(rgb_hsv_color_per)

        rgb_hsv_per    = []
        rgb_sorted     = []
        hsv_sorted     = []

        for ind in range(-1, -(len(rgb_colors) + 1), -1):
            rgb_hsv_per.append(rgb_hsv_color_per[rgb_hsv_sorted_ind[ind]])
            rgb_sorted.append(tuple(rgb_colors[rgb_hsv_sorted_ind[ind]]))
            hsv_sorted.append(tuple(hsv_colors[rgb_hsv_sorted_ind[ind]]))


        return rgb_sorted, hsv_sorted, rgb_hsv_per 

    except:
        return [(255, 255, 255)]*3
    

def check_group_range(grp1, grp2):
    
    h_threshold = 5
    s_threshold = 20
    v_threshold = 20
    
    p_h = 0.4
    p_s = 0.3
    p_v = 0.3
    
    if (abs(grp1[0] - grp2[0]) <= h_threshold) and \
    (abs(grp1[1] - grp2[1]) <= s_threshold) and \
    (abs(grp1[2] - grp2[2]) <= v_threshold):
        
        penalty = 1 - (((abs(grp1[0] - grp2[0]) * p_h) / 180.0) + \
        ((abs(grp1[1] - grp2[1]) * p_s) / 255.0 ) + \
        ((abs(grp1[2] - grp2[2]) * p_v) / 255.0 )) * 1.5
        
        return True, penalty

    else:
        return False, 0
    

def find_possible_ads(img, product_info, database_info):
    
    rgb_sorted_vid, hsv_sorted_vid, rgb_hsv_per_vid  = get_colors(img, 4, show_chart=True)
    
    similar_color = {}
    hsv_ads  = {}
    names = []
    
    category      = product_info['category']
    gender        = product_info['features']['gender']
    
   
    category_file = database_info.loc[(database_info['final_category']==f"['{category}']") & (database_info['final_gender']==f"['{gender}']")]

    for index, row in tqdm(category_file.iterrows()):
        #print(eval(row['hsv']))
        hsv_ads = [np.array(i) for i in eval(row['hsv'])]
        rgb_colors_ads, hsv_colors_ads, rgb_hsv_color_per_ads = merge_related_group(eval(row['rgb']),hsv_ads, eval(row['percent']), 4)
        per_add = 0
    
        len_ads = min(len(rgb_colors_ads), len(rgb_sorted_vid))
        
        for i in range(len_ads):
            
            flag, penalty = check_group_range(hsv_sorted_vid[i], hsv_colors_ads[i])
            if flag:
                per_to_add = min(rgb_hsv_color_per_ads[i], rgb_hsv_per_vid[i])
                per_add += per_to_add * penalty
        

        similar_color[row['ads_id']] = per_add

    final_dataframe = pd.DataFrame(columns=['ads_id', 'final_bbox'])
    for c, v in zip(similar_color.keys(), similar_color.values()):
        if v > 70:
            final_dataframe = final_dataframe.append(database_info[database_info['ads_id']==c][['ads_id', 'final_bbox']], ignore_index = True)
                
                
    return final_dataframe

