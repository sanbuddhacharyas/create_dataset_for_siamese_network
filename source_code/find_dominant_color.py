from sklearn.cluster import KMeans
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import sys


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def get_colors(image, number_of_colors=4, show_chart=False):
    
    try:
        modified_image = cv2.resize(image, (100, 100), interpolation = cv2.INTER_AREA)
        modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
        modified_image = np.where(modified_image <= 252, modified_image, 255)

        modified_image = modified_image[(modified_image!=np.array([[255, 255, 255]])).all(axis=1)]

        if list(modified_image)==[]:
            return (255, 255, 255)

        clf    = KMeans(n_clusters = number_of_colors)
        labels = clf.fit_predict(modified_image) 

        # sort to ensure correct color percentage
        counts = dict(sorted(Counter(labels).items()))

        center_colors = clf.cluster_centers_

        # We get ordered colors by iterating through the keys
        ordered_colors = [center_colors[i] for i in counts.keys()]
        rgb_colors     = [ordered_colors[i] for i in counts.keys()]

        dominant_colour = rgb_colors[np.argmax(list(counts.values()))]

        # if (show_chart):
        #     hex_colors     = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        #     print('make_figure')
        #     plt.figure(figsize = (8, 6))
        #     plt.pie(list(counts.values()), labels = hex_colors, colors = hex_colors)
        #     plt.show()


        return [int(i) for i in dominant_colour] 
    
    except:
        return (255, 255, 255)
        