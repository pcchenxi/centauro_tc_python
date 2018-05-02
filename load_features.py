import numpy as np
import cv2

from tool_functions.feature_fuser import *

feature_path = []
rgb_path = []
folder_path = '/home/xi/centauro_img/'
feature_file_list = os.listdir(folder_path)
for file_name in feature_file_list:
    if file_name.find('.npy') != -1:
        # print(file_name[:-13])
        rgb_p = folder_path + file_name[:-13] + '.jpg'
        path = file_name
        feature_path.append(folder_path + path)
        feature_path.append(rgb_p)
print('feature loaded:', len(feature_path))

def load_feature_file():
    count = 0
    for feature_file in feature_path:
        features_map = np.load(feature_file)
        count += 1
        # rgb_img = cv2
        print(count)
        # index = [0, 1, 2, 3, 4, 39]
        index = [1]
        for i in index:
            hd_img = features_map[:,:,i]
            # hd_img = show_img(hd_img, 'hd_img', 0)
            cv2.imshow('img', hd_img)
            cv2.waitKey(0)

load_feature_file()