import numpy as np
import cv2

from tool_functions.feature_fuser import *

feature_path = []
rgb_path = []
true_label_path = []

folder_path = '/home/xi/centauro_img/'
feature_file_list = os.listdir(folder_path)
for file_name in feature_file_list:
    if file_name.find('.npy') != -1:
        # print(file_name[:-13])
        rgb_p = folder_path + file_name[:-13] + '.jpg'
        label_p = folder_path + file_name[:-13] + '_label.png'
        path = file_name

        feature_path.append(folder_path + path)
        rgb_path.append(rgb_p)
        true_label_path.append(label_p)

print('feature loaded:', len(feature_path))

def process_feature(f_img):
    result_img = np.full(f_img.shape, 0, dtype=np.uint8)
    for row in range(f_img.shape[0]):
        for col in range(f_img.shape[1]):
            f_v = f_img[row, col]
            if f_v > 0.2:
                result_img[row, col] = 255

    return result_img

def load_feature_file():
    count = 0
    all_features = []
    all_labels = []

    for (feature_file, rgb_file, l_file) in zip(feature_path, rgb_path, true_label_path):
        print(feature_file)
        features_map = np.load(feature_file)
        rgb_img = cv2.imread(rgb_file, 1)
        true_label_img = cv2.imread(l_file, 0)
        count += 1

        # features = features_map[:,:,:-1]
        # labels =   features_map[:,:,-1]
    
        # shape = features.shape
        # features_reshaped = features.reshape(shape[0]*shape[1], shape[2])

        for row in range(features_map.shape[0]):
            for col in range(features_map.shape[1]):
                feature = features_map[row, col, :-1]
                label = features_map[row, col, -1]

                all_features.append(feature)
                all_labels.append(label)
        
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)

        print(all_labels.shape, all_features.shape)
        break


load_feature_file()