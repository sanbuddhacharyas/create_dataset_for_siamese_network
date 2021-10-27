import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from joblib import dump, load
import os
import json
import cv2
import requests
import urllib
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model
from PIL import Image
import io
from rembg.bg import remove
import tensorflow as tf

#Limit the GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
        
img_h , img_w = 256, 256

# METHOD #1: OpenCV, NumPy, and urllib
def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp  = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # return the image
    return image


def image_crop(img, annotation):
    W = img.shape[1]
    H = img.shape[0]
    
    x1 = annotation[0]
    y1 = annotation[1]
    x2 = annotation[2]
    y2 = annotation[3]
    

    return img[y1:y2, x1:x2, :]
    
def create_knn_model(database_info, product_info, num_group=180, save=True, hsv=False):
    
    category      = product_info['category']
    gender        = product_info['features']['gender']
    
    category_file = database_info.loc[(database_info['final_category']==f"['{category}']") & (database_info['final_gender']==f"['{gender}']")]
    
    if category_file.shape[0]==0:
        print("Items doesnot matches")
        return None, None, None, None, None, False
    
    if len(category_file) < num_group:
        num_group = int(len(category_file) * 0.5)
        
    rgb_colors    = category_file['final_RGB_ColorThief'].to_list()
    hsv_colors    = category_file['final_hsv'].to_list()
    colors        = np.array([eval(i)[0] for i in rgb_colors])
    product_color =  np.array([product_info['features']['r'], product_info['features']['g'],
              product_info['features']['b']])[np.newaxis,...]
    color_mode    = 'rgb'
    
    if hsv==True:
        print('HSV mode')
        colors    = np.array([eval(i)[0] for i in hsv_colors])
        product_color =  np.array([product_info['features']['h'], product_info['features']['s'],
              product_info['features']['v']])[np.newaxis,...]
        
        color_mode = 'hsv'
    
    model_file_name = f'model/{category}/{gender}/knn_{category}_{gender}_{color_mode}.joblib'
    
    all_images = np.concatenate([product_color, colors], axis=0)
    
    
    clf    = KMeans(n_clusters = num_group)
    clf.fit_predict(all_images)
    
    labels = clf.labels_
    mask   = (labels==labels[0])
    mask[0] = False
    indices = np.where(mask)
    indices = list(indices[0]-1)
    print(indices)
    
    neighbors = 20
    

    selected_items         = category_file.iloc[indices]
    selected_category      = selected_items[['ads_id','thumbnail_url', 'final_bbox']]
    
    if selected_items.shape[0]==0:
        print('Item Not found')
        return None, None, None, None, None, False
    
    selected_rgb           = np.concatenate([product_color, colors[indices]], axis=0)
    print(selected_rgb.shape)
    cos_indices, sim_score = cos_similarity(selected_rgb, neighbors)
    cos_selected_category  = selected_items.iloc[cos_indices][['ads_id','thumbnail_url', 'final_bbox']]
    
    if selected_items.shape[0]<neighbors:
        neighbors = selected_items.shape[0]
        
    knn = NearestNeighbors(n_neighbors=neighbors).fit(colors[indices])
    knn_score, knn_indices = knn.kneighbors(product_color)
    knn_selected_cateogry  = selected_items.iloc[knn_indices[0]][['ads_id','thumbnail_url', 'final_bbox']]
    
    intersected_product    = pd.merge(cos_selected_category, knn_selected_cateogry, how ='inner', on =['ads_id', 'thumbnail_url', 'final_bbox'])
    
    
    return category_file, selected_category, cos_selected_category, knn_selected_cateogry, intersected_product, True



def cos_similarity(concat_output, num_closest):

    similarity    = cosine_similarity(concat_output)# compute cosine similarities between images
    similarity_pd = pd.DataFrame(similarity, columns=range(len(concat_output)), index=range(len(concat_output)))

    sim = similarity_pd[0].sort_values(ascending=False)[1:num_closest+1].index
    sim_score = similarity_pd[0].sort_values(ascending=False)[1:num_closest+1].to_list()

    return np.array(sim) - 1, sim_score

def find_best_img(feature_vector, file_df,  num_best_outputs):
    
    cos_indices, sim_score = cos_similarity(feature_vector, num_best_outputs)
    cos_selected_category  = file_df.iloc[cos_indices][['ads_id','thumbnail_url']]
    return cos_selected_category, sim_score
    
def crop_images_boarder(img, width_ratio=0.4, u_height_ratio=0.3, l_heigh_ratio=0.3):
    
    img_ht, img_wt, layers = img.shape
    x , y = int(img_wt / 2), int(img_ht/ 2)

    x1 = int(x - width_ratio * img_wt)
    x2 = int(x + width_ratio * img_wt)
    y1 = int(y - u_height_ratio * img_ht)
    y2 = int(y + l_heigh_ratio * img_ht)
    
    return img[y1:y2, x1:x2]

def find_feature_vector(feature_extraction_model, detected_img, intersected_product, source_file, from_url=True):
    
    detected_img  = cv2.resize(detected_img, (img_w, img_h))   
    input_images = [detected_img]
    
    if from_url:
        
        
        for index, row in intersected_product.iterrows():
            img    = cv2.cvtColor(cv2.imread(f"{source_file}/{row['ads_id']}.png", cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            pil_im = Image.fromarray(img)
            byt = io.BytesIO()
            pil_im.save(byt, 'PNG')
            f_value = byt.getvalue()
            result_im = remove(f_value)
            imgs = Image.open(io.BytesIO(result_im))
            imgs.load() 

            background = Image.new("RGB", imgs.size, (255, 255, 255))
            background.paste(imgs, mask=imgs.split()[3]) # 3 is the alpha channel
            img = np.asarray(background)

            annotation = eval(row['final_bbox'])[0]
            crop_image = image_crop(img, annotation)
            crop_image = cv2.resize(crop_image,(img_w, img_h))   
            
            input_images.append(crop_image)
    
        input_images = np.stack(np.array(input_images), axis=0)
        feature_vector = feature_extraction_model.predict(input_images)
        

        return feature_vector
        
    else:
        
        for index, row in intersected_product.iterrows():
            
            try:
                img = url_to_image(row['thumbnail_url'])
                
            except:
                print("IMG URL ERROR==>", img_url)
                continue
            
            pil_im = Image.fromarray(img)
            byt = io.BytesIO()
            pil_im.save(byt, 'PNG')
            f_value = byt.getvalue()
            result_im = remove(f_value)
            imgs = Image.open(io.BytesIO(result_im))
            imgs.load() 

            background = Image.new("RGB", imgs.size, (255, 255, 255))
            background.paste(imgs, mask=imgs.split()[3]) # 3 is the alpha channel
            img = np.asarray(background)

            annotation = eval(row['final_bbox'])[0]
            
            print(annotation)
            print(img.shape)
            crop_image = image_crop(img, annotation)
            print(crop_image.shape)
            crop_image = cv2.resize(crop_image, (img_w, img_h))   
            
            input_images.append(crop_image)
    
        input_images = np.stack(np.array(input_images), axis=0)
        feature_vector = feature_extraction_model.predict(input_images)
        
        return feature_vector