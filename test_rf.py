
# coding: utf-8

# In[1]:


import numpy as np
import cv2
from matplotlib import pyplot as plt
import pickle

from tool_functions.feature_fuser import *


def process_feature(f_img):
    result_img = np.full(f_img.shape, 0, dtype=np.uint8)
    for row in range(f_img.shape[0]):
        for col in range(f_img.shape[1]):
            f_v = f_img[row, col]
            if f_v > 0.2:
                result_img[row, col] = 255

    return result_img


# In[2]:


def load_feature_file():
    count = 0
    train_features = []
    train_labels = []

    test_features = []
    test_labels = []
    
#     test_index = np.random.choice(29, 5, replace=False)
    test_files = range(29)
    # test_files = [2]
    test_index = []
    
    
    for (feature_file, rgb_file, l_file) in zip(feature_path, rgb_path, true_label_path):
        print("in load file", feature_file)
        features_map = np.load(feature_file)
        h_img_file = feature_file[0:-4]
        h_img_file = h_img_file + '_stair.png'

        h_feature = cv2.imread(rgb_file, 0)
        features_map[:,:,4] = h_feature

        rgb_img = cv2.imread(rgb_file, 1)
        true_label_img = cv2.imread(l_file, 0)/50
        
        img_index = rgb_file[len(folder_path):-8]
#         print(rgb_file[len(folder_path):-8])
        
        is_train = True
        for i in range(len(test_files)):
            if img_index == str(test_files[i]):
                is_train = False
                test_index.append(count)
                print(is_train, feature_file, features_map.shape, true_label_img.shape, count)
                break
            
#         print(is_train, feature_file, features_map.shape, true_label_img.shape, count)
        count += 1

        # features = features_map[:,:,:-1]
        # labels =   features_map[:,:,-1]
    
        # shape = features.shape
        # features_reshaped = features.reshape(shape[0]*shape[1], shape[2])

        features_map_reshape = features_map.reshape((features_map.shape[0]*features_map.shape[1], features_map.shape[2]))
        label_reshape = true_label_img.reshape(true_label_img.shape[0]*true_label_img.shape[1])
        if is_train:
            if len(train_features) == 0:
                train_features = features_map_reshape
                train_labels = label_reshape
            else:
                train_features = np.concatenate((train_features,features_map_reshape), axis=0)
                train_labels = np.concatenate((train_labels,label_reshape), axis=0)
            
#             train_features.append(features_map_reshape)
#             train_labels.append(label_reshape)      

#         for row in range(size, features_map.shape[0]-size, 1+size):
#             for col in range(size, features_map.shape[1]-size, 1+size):
#                 feature = features_map[row-size:row+size+1, col-size:col+size+1].flatten()
#                 label = int(true_label_img[row, col]/50)
#                 if label != -1 and label != 0:
#                     if is_train:
#                         train_features.append(feature)
#                         train_labels.append(label)
#                     else:
#                         test_features.append(feature)
#                         test_labels.append(label)
        
    train_features = np.array(train_features)
    train_labels = np.array(train_labels)
#     test_features = np.array(test_features)
#     test_labels = np.array(test_labels)
    
    print(train_features.shape, train_labels.shape)

    return train_features, train_labels, test_index


# In[3]:


# In[7]:


def classify_features(features_map, true_label_img, rgb_img):
    pred_img = np.full((540, 960, 3), 0, np.uint8)
    true_img = np.full((540, 960, 3), 0, np.uint8)

    features_map_reshape = features_map.reshape((features_map.shape[0]*features_map.shape[1], features_map.shape[2]))
    feature_nromalized = scaler.transform(features_map_reshape)
    print('predicting.....')
    label_pred = clf.predict(feature_nromalized)

#     true_label = true_label_img.reshape(true_label_img.shape[0]*true_label_img.shape[1])/50
#     score = clf.score(feature_nromalized, true_label)
#     print ('score', score)
    
    print('predected!', label_pred.shape)
    label_pred_reshaped = label_pred.reshape((features_map.shape[0], features_map.shape[1]))

    total_count = 0
    correct_count = 0
    
    for row in range(features_map.shape[0]):
        for col in range(features_map.shape[1]):
            label_pred = label_pred_reshaped[row, col]
            true_label = true_label_img[row, col]/50
            
            if true_label == 5: #narrow passage
                true_img[row, col] = [255, 255, 0]
            if true_label == 4: #stair
                true_img[row, col] = [255, 255, 0]
            if true_label == 3: #obs
                true_img[row, col] = [0, 0, 255]     
            if true_label == 2: #rough
                true_img[row, col] = [0, 255, 0]  
            if true_label == 1: #safe
                true_img[row, col] = [255, 0, 0] 
                    
                    
                    
            if features_map[row, col, 0] != -1:
                total_count += 1
                if label_pred == true_label:
                    correct_count += 1
                
                if label_pred == 5: #narrow passage
                    cv2.circle(pred_img, (col, row), 3, (255, 255, 0), -1)                
                if label_pred == 4: #stair
                    cv2.circle(pred_img, (col, row), 5, (255, 255, 0), -1)
                if label_pred == 3: #obs
                    cv2.circle(pred_img, (col, row), 3, (0, 0, 255), -1)     
                if label_pred == 2: #rough
                    cv2.circle(pred_img, (col, row), 3, (0, 255, 0), -1)  
                if label_pred == 1: #safe
                    cv2.circle(pred_img, (col, row), 3, (255, 0, 0), -1) 

    print(correct_count, total_count, correct_count/total_count)
    return pred_img, true_img, correct_count/total_count


# In[10]:

feature_path = []
rgb_path = []
true_label_path = []
size = 0

folder_path = '/home/xi/workspace/bonn_features/'
feature_file_list = os.listdir(folder_path)
for file_name in feature_file_list:
    if file_name.find('.npy') != -1:
        # print(file_name[:-13])
        rgb_p = folder_path + file_name[:-4] + '_img.jpg'
        label_p = folder_path + file_name[:-4] + '_img_label.png'
        path = file_name

        feature_path.append(folder_path + path)
        rgb_path.append(rgb_p)
        true_label_path.append(label_p)


print('feature loaded:', len(feature_path))
# print(true_label_path)


train_features, train_labels, test_index = load_feature_file()
print(test_index)


# In[4]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(train_features)

X_train = scaler.transform(train_features)
# X_test = scaler.transform(test_features)

y_train = train_labels
# y_test = test_labels

# X = StandardScaler().fit_transform(all_features)
# X_train, X_test, y_train, y_test = train_test_split(all_features, all_labels, test_size=.4, random_state=42)


# In[5]:


from sklearn.ensemble import RandomForestClassifier
# print('training...')
# clf = RandomForestClassifier(max_depth=30, n_estimators=10, max_features=1)
# clf.fit(X_train, y_train)

# pickle.dump(clf, open("/home/xi/workspace/terrain_classifier/tc_model.pickle","wb"), protocol=2)
# pickle.dump(scaler, open("/home/xi/workspace/terrain_classifier/tc_model_scaler.pickle","wb"), protocol=2)

# print("done!!")

clf = pickle.load(open('/home/xi/workspace/terrain_classifier/tc_model.pickle', 'rb'))
scaler = pickle.load(open('/home/xi/workspace/terrain_classifier/tc_model_scaler.pickle', 'rb'))

sum_score = 0
for i in range(len(test_index)):

    index = test_index[i]
    print(index)
#     index = 1
    feature_file = feature_path[index]
    rgb_file = rgb_path[index]
    l_file = true_label_path[index]
    true_label_img = cv2.imread(l_file, 0)
    
    print(feature_file)
    features_map = np.load(feature_file)
    rgb_img = cv2.imread(rgb_file, 1)

    pred_img, true_img, score = classify_features(features_map, true_label_img, rgb_img)
    h_stair = features_map[:, :, 4]*255
    
    sum_score += score
#     print(true_label_img)
#     print(pred_img)
    
    test_size = len(test_index)
    index = 2
    file_base = "/home/xi/bonn_images/"+feature_file[33:-4]
    print(file_base)
    cv2.imwrite(file_base +'_img.png', rgb_img)
    cv2.imwrite(file_base +'_label.png', true_img)
    cv2.imwrite(file_base +'_pred.png', pred_img)
    cv2.imwrite(file_base +'_stair.png', h_stair)
    
    # f = plt.figure()
    # f.add_subplot(1,3, 1)
    # plt.imshow(rgb_img)
    # f.add_subplot(1,3, 2)
    # plt.imshow(h_stair)   
    # f.add_subplot(1,3, 3)
    # plt.imshow(pred_img)      
    # plt.show()  
    
print('done')
print (sum_score/len(test_index))
    
    
  
                

