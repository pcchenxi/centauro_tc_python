import sys, time, cv2, pcl
import numpy as np


Map_W = 16
Map_B = 16
Map_resolution = 0.08

def show_img(img, name, waitkey, use_mask=True):
    # print(name, img.min(), img.max()) 

    img_max = img.max()
    img_min = img.min()

    img = (img - img_min)
    img = img/img_max * 255
    # print(img.min(), img.max())    
    img = img.astype(np.uint8)

    if use_mask:
        cv2.imshow(name, img*g_zero_mask)
    else:
        cv2.imshow(name, img)
    cv2.waitKey(waitkey)
    # print(random_image)
    # random_image = random_image.astype(np.uint8)
    # print(img.min(), img.max())    
    # plt.clf()
    # plt.imshow(random_image, cmap='gray')
    # plt.pause(0.01)
    

def filter_cloud_by_minh(cloud):
    global g_zero_mask

    init_h = 0
    scale = 0.3
    min_img = np.full((int(Map_W/Map_resolution), int(Map_B/Map_resolution)), init_h, np.float32)
    min_img_small = np.full((int(Map_W/Map_resolution * scale), int(Map_B/Map_resolution * scale)), init_h, np.float32)
    
    point_row_col = []
    point_row_col_small = []

    # print(min_img.shape)
    for p in cloud:
        row = int((p[0]+Map_W/2)/Map_resolution)
        col = int((p[1]+Map_W/2)/Map_resolution)

        row_small = int((p[0]+Map_W/2)/Map_resolution * scale)
        col_small = int((p[1]+Map_W/2)/Map_resolution * scale)

        point_row_col.append((row, col))
        point_row_col_small.append((row_small, col_small))

        value_pre = min_img[row, col]
        if value_pre > p[2] or value_pre == init_h:
            min_img[row, col] = p[2]

        value_small_pre = min_img_small[row_small, col_small]
        if value_small_pre > p[2]:
            min_img_small[row_small, col_small] = p[2]

    # extract hight difference
    max_img_f = np.full((int(Map_W/Map_resolution), int(Map_B/Map_resolution)), init_h, np.float32)
    min_img_f = np.full((int(Map_W/Map_resolution), int(Map_B/Map_resolution)), init_h, np.float32)

    sumh_img = np.full((int(Map_W/Map_resolution), int(Map_B/Map_resolution)), 0, np.float32)
    count_img = np.full((int(Map_W/Map_resolution), int(Map_B/Map_resolution)), -1, np.float32)

    point_row_col_f = []
    filtered_cloud = []

    for p, (row, col), (row_small, col_small) in zip(cloud, point_row_col, point_row_col_small):
        # min_value = min_img[row, col]
        min_value_small = min_img_small[row_small, col_small]

        if p[2] < min_value_small + 1:
            filtered_cloud.append(p)
            point_row_col_f.append((row, col))

            max_pre = max_img_f[row, col]
            min_pre = min_img_f[row, col]
            if max_pre < p[2] or max_pre == -init_h:
                max_img_f[row, col] = p[2]
            if min_pre > p[2] or min_pre == init_h:
                min_img_f[row, col] = p[2]   

            sumh_img[row, col] += p[2] 
            if count_img[row, col] == -1:
                count_img[row, col] = 1
            else:
                count_img[row, col] += 1

    h_img = sumh_img/count_img    


    points_filtered_by_h = []
    for p, (row, col), (row_small, col_small) in zip(cloud, point_row_col, point_row_col_small):
        h_value = h_img[row, col]
        points_filtered_by_h.append([p[0], p[1], h_value])

    # pcl_filtered_by_h = pcl.PointCloud()
    # pcl_filtered_by_h.from_list(points_filtered_by_h)
    # publish_cloud(pcl_filtered_by_h, pcl_pub)   

    zero_mask = np.zeros_like(min_img_f)
    g_zero_mask = (min_img_f==zero_mask)
    g_zero_mask = 1 - g_zero_mask.astype(np.uint8)

    # show_img(abs(max_img_f - min_img_f), 'diff', 10)
    # show_img(max_img_f, 'max', 10)
    show_img(h_img, 'min', 10)

    pcl_data = pcl.PointCloud()
    pcl_data.from_list(filtered_cloud)

    return pcl_data, point_row_col_f, abs(max_img_f-min_img_f)

def compute_normal(cloud, radius):
    ne = cloud.make_NormalEstimation()
    ne.set_RadiusSearch(radius)
    normals = ne.compute()

    return normals.to_array()

def detect_narrow(img):
    size = int(0.8/Map_resolution)
    kernel_dilate = np.ones((size,size),np.uint8)
    kernel_clean = np.ones((3,3),np.uint8)

    close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_dilate)

    # dilation = cv2.dilate(img, kernel_dilate,iterations = 1)
    # dilation_diff = cv2.dilate(img, kernel_diff,iterations = 1)
    # erosion = cv2.erode(dilation, kernel_dilate,iterations = 1)
    # erosion = cv2.erode(dilation,kernel_dilate,iterations = 1)

    img_diff = (close - img)
    narrow = cv2.erode(img_diff,kernel_clean,iterations = 1)    

    # cv2.imshow('obs', img)
    # cv2.imshow('dilation', dilation)
    # cv2.imshow('dilation_diff', dilation_diff)
    # cv2.imshow('erosion', erosion)
    # cv2.imshow('img_diff', img_diff)
    # cv2.waitKey(0)

    return narrow


def detect_narrow_passage(hdiff_img):
    _, obs_img = cv2.threshold(hdiff_img, 0.3, 255, cv2.THRESH_BINARY)

    size = int(0.4/Map_resolution)
    kernel = np.ones((size,size),np.uint8)
    obs_img = cv2.dilate(obs_img,kernel,iterations = 1)

    narrow_mask = detect_narrow(obs_img)
    narrow_mask = narrow_mask*g_zero_mask 

    obs_rgb = cv2.cvtColor(obs_img,cv2.COLOR_GRAY2RGB)
    obs_rgb[:,:,2] = 0
    obs_rgb[:,:,2] = narrow_mask

    cv2.imshow('obs_rgb', obs_rgb)
    return obs_rgb

def get_slope_roughness_img(cloud_filtered, point_row_col):
    ################################################################################################
    # compute slope and roughness
    slope_img = np.full((int(Map_W/Map_resolution), int(Map_B/Map_resolution)), -1, np.float32)
    roughness_img = np.full((int(Map_W/Map_resolution), int(Map_B/Map_resolution)), -1, np.float32)
    normal_big = compute_normal(cloud_filtered, Map_resolution*4)
    normal_small = compute_normal(cloud_filtered, Map_resolution*2)

    sumh_slope = np.full((int(Map_W/Map_resolution), int(Map_B/Map_resolution)), 0, np.float32)
    sumh_roughness = np.full((int(Map_W/Map_resolution), int(Map_B/Map_resolution)), 0, np.float32)
    count_img = np.full((int(Map_W/Map_resolution), int(Map_B/Map_resolution)), 1, np.float32)

    for n_b, n_s, (row, col)in zip(normal_big, normal_small, point_row_col):
        if np.isnan(n_b[2]) or np.isnan(n_s[2]):
            continue
        # min_value = min_img[row, col]
        # print(n_b[2], n_s[2])
        slope = 1 - abs(n_b[2])
        roughness = abs(abs(n_b[2]) - abs(n_s[2]))

        sumh_slope[row, col] += slope
        sumh_roughness[row, col] += roughness
        count_img[row, col] += 1

        # slope_pre = slope_img[row, col]
        # roughness_pre = roughness_img[row, col]

        # if slope_pre < slope or slope_pre == -1:
        #     slope_img[row, col] = slope
        # if roughness_pre < roughness or roughness_pre == -1:
        #     roughness_img[row, col] = roughness

    slope_img = sumh_slope/count_img
    roughness_img = sumh_roughness/count_img

    show_img(slope_img, 'slope_img', 10)
    show_img(roughness_img, 'roughness_img', 10)

    return slope_img, roughness_img

def compute_geometric_features(map_cloud_pcl, map_w, map_b, map_resolution):
    global Map_W, Map_B, Map_resolution

    Map_W = map_w 
    Map_B = map_b 
    Map_resolution = map_resolution

    cloud_filtered, point_row_col, hdiff_img = filter_cloud_by_minh(map_cloud_pcl)
    show_img(hdiff_img, 'hdiff_img', 10)

    obs_rgb = detect_narrow_passage(hdiff_img)

    ###############################################
    # compute slope and roughness
    ###############################################
    slope_img, roughness_img = get_slope_roughness_img(cloud_filtered, point_row_col)    

    return hdiff_img, slope_img, roughness_img, point_row_col