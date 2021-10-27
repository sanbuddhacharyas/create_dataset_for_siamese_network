import urllib
import numpy as np
import cv2
from skimage.transform import resize
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model 
import pandas as pd
from glob import glob

class image_matching:   

    #If pre_trained model weights is availabe pass its weights path
    def __init__(self, pre_trained_model_path = None):
        self.pre_trained_model_path = pre_trained_model_path

    def url_to_image(self, url):
        # download the image, convert it to a NumPy array, and then read
        # it into OpenCV format
        resp  = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # return the image
        return image

     
    def feature_extractor_model(self):
        """ 
            this function creates VGG feature extraction model
            arg:
                None
            return:
                model: keras Model object

        """

        if self.pre_trained_model_path==None:
            input_tensor = Input((224, 224, 3))
            input_tensor = preprocess_input(input_tensor)
            base_model   = VGG19(input_tensor=input_tensor, weights = 'imagenet')
            model        = Model(inputs=base_model.input, outputs = base_model.get_layer("fc2").output)

            return model

        else:
            print(f'Loaded_model_from==>{self.pre_trained_model_path}')
            return load_model(self.pre_trained_model_path)


    
    def get_images_features(self, images_list):
        """
            Extract dense features of images 
            arg:
                images_list: list of images to extact features
            return:
                output: numpy array of extracted features
        """

        vgg19_model = self.feature_extractor_model()   #
        images_list = np.stack(images_list, axis=0)    #convert list into numpy with shape(None, H, W, C)
        output = vgg19_model.predict(images_list)   

        return output


    def find_best_img(self, img, images_features, num_closest):
        """
            Finds the matching score between the iamges
            arguments:
                img = (numpy array) image whose best matches to be find
                images_features = (numpy array) of features extracted from where best matching is to be find
                num_closest     = (int) number of similar images
            
        """

        vgg19_model = self.feature_extractor_model()
        img  = img[np.newaxis,...]     # download the image, convert it to a NumPy array, and then read
        output = vgg19_model.predict(img)         # predict output

        concat_output = np.concatenate([output, images_features], axis=0)
        similarity = cosine_similarity(concat_output)# compute cosine similarities between images
        similarity_pd = pd.DataFrame(similarity, columns=range(len(concat_output)), index=range(len(concat_output)))

        sim = similarity_pd[0].sort_values(ascending=False)[1:num_closest+1].index
        sim_score = similarity_pd[0].sort_values(ascending=False)[1:num_closest+1].to_list()

        return np.array(sim) - 1, sim_score

    