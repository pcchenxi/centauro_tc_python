import numpy as np
import pcl, cv2, os
from pcl import IterativeClosestPoint, GeneralizedIterativeClosestPoint, IterativeClosestPointNonLinear
from PIL import Image

current_path = []
current_path_project = []
rgb_img, label_img, h_label_img = [], [], []
img_path = [], [], []

img_path = []
folder_path = '/home/xi/centauro_img/'
feature_file_list = os.listdir(folder_path)
for file_name in feature_file_list:
    if file_name.find('.jpg') != -1:
        path = file_name
        img_path.append(path)
print('image loaded:', len(img_path))


def load_loabls(rosbag_name):
    global img_path, rgb_img, label_img, h_label_img, current_path, current_path_project

    base_path = '/home/xi/data_recorded/testing_data/'
    name = rosbag_name[len(base_path):-4]
    found_match = False

    # print('     looking for matched label.....')
    for i in range(len(img_path)):
        if img_path[i].find(name) != -1:
            rgb_img = cv2.imread(folder_path+img_path[i], 1)

            label_path = folder_path + img_path[i][:-4] + '_label.png'
            label_img = cv2.imread(label_path, 0)

            label_path = folder_path + img_path[i][:-4] + '.tiff'
            im = Image.open(label_path)
            h_label_img = np.asarray(im)

            current_path = folder_path + img_path[i][:-4] + '_features'
            current_path_project = folder_path + '3d/' + img_path[i][:-4] + '_3d.png'
            found_match = True
            break

    cv2.imshow('rgb', rgb_img)
    cv2.imshow('label', label_img)
    cv2.imshow('h_label_img', h_label_img)    
    cv2.waitKey(10)
    # print('     done', found_match)
    return found_match

def show_img(img, name, waitkey):
    print(name, img.min(), img.max()) 

    img_max = img.max()
    img_min = img.min()

    print(img_max, img_min)
    img = img-img_min
    img = img/img_max * 255
    img = img.astype(np.uint8)

    cv2.imshow(name, img)
    cv2.waitKey(waitkey)

    return img

def get_min_distimg(uvd_rc):
    min_dist_img = np.full((540, 960), 0, np.float32)
    for (u, v, dist, row, col) in uvd_rc:
        u, v = int(u), int(v)
        dist_pre = min_dist_img[v, u]
        if dist_pre == 0 or dist < dist_pre:
            min_dist_img[v, u] = dist

    # show_img(min_dist_img, 'min_dist_img', 0)    
    return min_dist_img

def save_all_features(hdiff_img, slope_img, roughness_img, uvd_rc, feature_vision):
    global img_path, rgb_img, label_img, h_label_img
    print('save all features')
    min_dist_img = get_min_distimg(uvd_rc)


    cnn_img_size = feature_vision.shape
    features_map = np.full((540, 960, 40), -1.0)
    test_img = np.full((540, 960), 0, np.float32)


    scale_row = cnn_img_size[1]/float(features_map.shape[0])
    scale_col = cnn_img_size[2]/float(features_map.shape[1])

    # print(scale_row, scale_col, features_map.shape, cnn_img_size)
    for (u, v, dist, row, col) in uvd_rc:
        u, v, row, col = int(u), int(v), int(row), int(col)

        min_dist = min_dist_img[v, u]
        if dist > min_dist:
            continue 
    
        v_cnn = int(v*scale_row)
        u_cnn = int(u*scale_col)

        # test_img[v, u] = label_img[v, u]/50
        color = int(label_img[v, u])
        cv2.circle(rgb_img, (u,v), 3, (color, color, color), -1)

        features_map[v, u, 0] = hdiff_img[row, col]
        features_map[v, u, 1] = slope_img[row, col]
        features_map[v, u, 2] = roughness_img[row, col]
        features_map[v, u, 3] = dist
        features_map[v, u, 4] = h_label_img[v, u]

        features_map[v, u, 5:39] = feature_vision[0, v_cnn, u_cnn, :]

        features_map[v, u, 39] = label_img[v, u]/50

        # print(hd, slope, roughness, stair_feature, len(cnn_features), label/50)

    hd_img = features_map[:,:,3]
    hd_img = show_img(hd_img, 'hd_img', 10)
    cv2.imshow('mapping', rgb_img)
    cv2.waitKey(10)

    file_name = current_path
    np.save(file_name, features_map)
    cv2.imwrite(current_path_project, rgb_img)
    # print(file_name)
