import numpy as np
import pcl, cv2
from pcl import IterativeClosestPoint, GeneralizedIterativeClosestPoint, IterativeClosestPointNonLinear


rgb_imgs, label_imgs, img_path = [], [], []

def rotate_map_cloud(map_cloud, kinect_cloud):
    print('start icp')
    icp = map_cloud.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(map_cloud, kinect_cloud, max_iter=1000)

    point_rotated = []
    for p in map_cloud:
        p_r = np.dot(transf, np.array([p[0], p[1], p[2], 1.0]))[:3]

        point_rotated.append(p_r)

    cloud_rotated = pcl.PointCloud()
    cloud_rotated.from_list(point_rotated)

    print('done icp', transf)
    return transf, cloud_rotated

def load_loabls():
    global rgb_imgs, label_imgs, img_path
    img_path = ['/home/xi/centauro_img/terrain_museum_side_1_0007.jpg', '/home/xi/centauro_img/terrain_museum_side_1_0006.jpg', \
         '/home/xi/centauro_img/terrain_museum_side_1_0005.jpg', '/home/xi/centauro_img/terrain_museum_side_1_0004.jpg', \
         '/home/xi/centauro_img/terrain_museum_side_1_0002.jpg', '/home/xi/centauro_img/terrain_museum_side_1_0000.jpg', \
         '/home/xi/centauro_img/stairs_universitetet_moving_1_0006.jpg', '/home/xi/centauro_img/stairs_universitetet_moving_1_0005.jpg', \
         '/home/xi/centauro_img/stairs_universitetet_moving_1_0004.jpg', '/home/xi/centauro_img/stairs_universitetet_moving_1_0003.jpg', \
         '/home/xi/centauro_img/stairs_universitetet_moving_1_0002.jpg', '/home/xi/centauro_img/stairs_universitetet_moving_1_0001.jpg', \
         '/home/xi/centauro_img/stairs_universitetet_moving_1_0000.jpg', '/home/xi/centauro_img/stairs_universitetet_5_0000.jpg', \
         '/home/xi/centauro_img/stairs_universitetet_4_0000.jpg', '/home/xi/centauro_img/stairs_universitetet_3_0000.jpg',\
         '/home/xi/centauro_img/stairs_universitetet_2_0000.jpg', '/home/xi/centauro_img/stairs_universitetet_1_0000.jpg', \
         '/home/xi/centauro_img/stairs_museum_side_5_0000.jpg', '/home/xi/centauro_img/stairs_museum_side_4_0000.jpg', \
         '/home/xi/centauro_img/stairs_museum_side_3_0000.jpg', '/home/xi/centauro_img/stairs_museum_side_2_0000.jpg', \
         '/home/xi/centauro_img/stairs_museum_side_1_0000.jpg', '/home/xi/centauro_img/terrain_museum_behind_moving_0008.jpg', \
         '/home/xi/centauro_img/terrain_museum_behind_moving_0007.jpg', '/home/xi/centauro_img/terrain_museum_behind_moving_0006.jpg', \
         '/home/xi/centauro_img/terrain_museum_behind_moving_0005.jpg', '/home/xi/centauro_img/terrain_museum_behind_moving_0004.jpg', \
         '/home/xi/centauro_img/terrain_museum_behind_moving_0003.jpg', '/home/xi/centauro_img/terrain_museum_behind_moving_0002.jpg', \
         '/home/xi/centauro_img/terrain_museum_behind_moving_0001.jpg', '/home/xi/centauro_img/terrain_museum_behind_moving_0000.jpg', \
         '/home/xi/centauro_img/stairs_museum_front_5_0000.jpg', '/home/xi/centauro_img/stairs_museum_front_4_0000.jpg', \
         '/home/xi/centauro_img/stairs_museum_front_3_0000.jpg', '/home/xi/centauro_img/stairs_museum_front_2_0000.jpg', \
         '/home/xi/centauro_img/stairs_museum_front_1_0000.jpg', '/home/xi/centauro_img/stairs_museum_down_3_0000.jpg', \
         '/home/xi/centauro_img/stairs_museum_down_2_0000.jpg', '/home/xi/centauro_img/stairs_museum_behind_2_0000.jpg', \
         '/home/xi/centauro_img/stairs_museum_behind_1_0000.jpg', '/home/xi/centauro_img/stairs_lappis_7_0000.jpg', \
         '/home/xi/centauro_img/stairs_lappis_6_0000.jpg', '/home/xi/centauro_img/stairs_lappis_5_0000.jpg', \
         '/home/xi/centauro_img/stairs_lappis_4_0000.jpg', '/home/xi/centauro_img/stairs_lappis_2_0000.jpg', \
         '/home/xi/centauro_img/stairs_opposite_tek14_8_0000.jpg', '/home/xi/centauro_img/stairs_tek31_2_0001.jpg', \
         '/home/xi/centauro_img/stairs_behind_tek14_down_3_0000.jpg', '/home/xi/centauro_img/stairs_afterb_1_0000.jpg', \
         '/home/xi/centauro_img/stairs_tek78_2_0000.jpg', '/home/xi/centauro_img/stairs_tek78_1_0000.jpg', \
         '/home/xi/centauro_img/stairs_tek31_2_0000.jpg', '/home/xi/centauro_img/stairs_tek31_0000.jpg', \
         '/home/xi/centauro_img/stairs_tek14_side_0000.jpg', '/home/xi/centauro_img/stairs_tek14_down_2_0000.jpg', \
         '/home/xi/centauro_img/stairs_tek14_down_1_0000.jpg', '/home/xi/centauro_img/stairs_tek14_2_0000.jpg', \
         '/home/xi/centauro_img/stairs_tek14_1_0000.jpg', '/home/xi/centauro_img/stairs_tek10_3_0000.jpg', \
         '/home/xi/centauro_img/stairs_osqbacke_2_0000.jpg', '/home/xi/centauro_img/stairs_opposite_tek14_10_0000.jpg', \
         '/home/xi/centauro_img/stairs_opposite_tek14_9_0000.jpg', '/home/xi/centauro_img/stairs_opposite_tek14_6_0000.jpg', \
         '/home/xi/centauro_img/stairs_opposite_tek14_5_0000.jpg', '/home/xi/centauro_img/stairs_opposite_tek14_3_0000.jpg', \
         '/home/xi/centauro_img/stairs_opposite_tek14_2_0000.jpg', '/home/xi/centauro_img/stairs_kthhallen_1_0000.jpg', \
         '/home/xi/centauro_img/stairs_brinel23_1_0000.jpg']


    for path in img_path:
        img = cv2.imread(path, 1)
        rgb_imgs.append(img)

        label_path = path[:-4] + '_label.png'
        img_label = cv2.imread(label_path, 0)
        label_imgs.append(img_label)

    print('loaded', len(rgb_imgs), len(label_imgs))

def match_img(string_p):    
    # img = cv2.cvtColor( img, cv2.COLOR_RGB2GRAY )
    # cv2.imshow('img', img)
    # cv2.waitKey(10)

    match_index = -1
    min_diff = -1
    print('matching db img', string_p)

    for i in range(len(img_path)):
        if img_path[i].find(string_p) != -1:
            match_index = i 
            break
    # for i in range(len(rgb_imgs)):
        # diff_sum = np.sum(abs(rgb_imgs[i]-img).flatten())/100
        # if min_diff == -1 or diff_sum < min_diff:
        #     match_index = i 
        #     min_diff = diff_sum
        #     print(min_diff)

    if match_index == -1:
        return []

    cv2.imshow("matched", rgb_imgs[match_index])
    print('matched', match_index)
    cv2.waitKey(10)

    return label_imgs[match_index]
