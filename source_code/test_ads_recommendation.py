from find_analytical_json import VideoAnalytics
from ads_recomender  import create_knn_model, find_feature_vector, find_best_img, url_to_image, image_crop
from ads_recomender_multicolor import find_possible_ads, get_image

#import required libraies
import json
import pandas as pd
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from glob import glob
import requests
import urllib.request 
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

file_df = pd.read_csv('../updated_ads_repo_v12.csv')
filter_df = file_df.loc[file_df['Final Objects']!='[]']
ads_image_path = '../rembg_all_images'
img_h , img_w = 256, 256

def find_best_img_knn(feature_vector,  file_df,  num_best_outputs):
    
    print("Inside Knn")
    
    
    if (feature_vector.shape[0]-1) <= num_best_outputs:
        num_best_outputs = feature_vector.shape[0]-1
        print(num_best_outputs)
        
    knn          = NearestNeighbors(n_neighbors=num_best_outputs).fit(feature_vector[1:])
    score, index = knn.kneighbors(feature_vector[0].reshape(1,-1))
    print("outside Knn")    
    knn_selected_category  = file_df.iloc[index[0]][['ads_id', 'final_bbox']]
    
    return knn_selected_category, score[0]

def url_to_mp4(video_url, save_path):
    urllib.request.urlretrieve(video_url, save_path) 

def download_videos(download_list, save_location):
    
    with open(download_list, 'r') as f:
        video_to_download = f.readlines()
        
    list_of_videos = [i.split('/')[-1].replace('.mp4', '') for i in glob(save_location+'/*.mp4')]
    print(list_of_videos)
    
    print("Downloading videos")
    for video_url in tqdm(video_to_download):
        
        if video_url.split('/')[-1].replace('.mp4','').replace('\n', '') in list_of_videos:
            print(f"Alread_present=>{video_url.split('/')[-1].replace('.mp4','')}")
            continue
            
        else:
            save_location = '../test_videos/'+video_url.split('/')[-1].replace('\n', '')
            url_to_mp4(video_url, save_location)

            
def find_video_analytics(json_location, video_location):
    with open('video_analytics.txt', 'r') as f:
        already_json = f.readlines()
        
        already_json = [i.replace('\n', '') for i in already_json]
            
#     list_of_json   = [i.split('/')[-1].replace('.json', '') for i in glob(json_location + '/*.json')]
#     list_of_json   = []    
#     print(f"List_of_json==>{list_of_json}")
    video_location = glob(video_location+ '/*.mp4')
    
    print("Creating video analytics json")
    for video_path in tqdm(video_location):
        
            
        if video_path.split('/')[-1].replace('.mp4', '') in already_json:
#             print(f"Alread_present=>{video_path.split('/')[-1].replace('.mp4','')}")
            continue
         
        else:
            with open('video_analytics.txt', 'a') as f:
                f.write(video_path.split('/')[-1].replace('.mp4', '')+'\n')
            print(video_path)
            VideoAnalytics(video_path)
            
        

def find_best_ads(json_location, multicolor=True, num_output=3):

    with open(json_location, 'r') as f:
        file_json = json.load(f)
        
    for detected_product in range(len(file_json['analyticsInfo'])): 
        
        detected_image = cv2.imread(f"../detected_image/{file_json['analyticsInfo'][detected_product]['objectImage']}")
        detected_image = cv2.resize(cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB), (256,256))
            
        intersected_product = find_possible_ads(detected_image, file_json['analyticsInfo'][detected_product], filter_df)
        if intersected_product.shape[0]==0:
            continue
    
        category   = file_json['analyticsInfo'][detected_product]['category']
        video_name = file_json['analyticsInfo'][detected_product]['objectImage'].replace('.png','')
        os.makedirs(f"../dataset/{category}/{video_name}", exist_ok=True)
        
        cv2.imwrite(f"../dataset/{category}/{video_name}/original.png", cv2.cvtColor(detected_image, cv2.COLOR_RGB2BGR))
        
        for ind, row in intersected_product.iterrows():

            img = get_image(f"{ads_image_path}/{row['ads_id']}.png")
            img = image_crop(img, eval(row['final_bbox'])[0])
            img = cv2.cvtColor(cv2.resize(img, (img_w, img_h)), cv2.COLOR_RGB2BGR)

            cv2.imwrite(f"../dataset/{category}/{video_name}/{row['ads_id']}.png", img)



            
with open('../list_of_video_to_download.txt', 'r') as f:
    video_to_download = f.readlines()
    

json_location  = "../analytics_json"  
download_list  = "../list_of_video_to_download.txt"
video_location = '../test_videos'

download_videos(download_list, video_location)
find_video_analytics(json_location, video_location)

list_of_suggested_ads = [i.split('/')[-1] for i in glob('../Suggested_ads/*')]
list_of_json = glob(json_location+"/*.json")

print(f"List_of_suggested_ads==>{list_of_suggested_ads}")
for json_path in list_of_json:
    if json_path.split('/')[-1].replace('.json', '') in list_of_suggested_ads:
        continue
    find_best_ads(json_path)