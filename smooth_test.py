import tf as ros_tf
import rospy, sys, time
import pcl, cv2
import matplotlib.pyplot as plt
import std_msgs.msg
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud, PointCloud2, PointField
import numpy as np
import pcl_related_function
import geometry_msgs.msg
from sensor_msgs.msg import Image
import pickle
import tensorflow as tf
import scipy as sp
from cv_bridge import CvBridge, CvBridgeError
from feature_fuser import *

# for cnn
path = '/home/xi/workspace/catkin_centauro/src/fused_terrain_classifier/src/vision' #os.getcwd()
sys.path.append(path + '/segmentation/')
sys.path.append(path + '/segmentation/datasets/')
sys.path.append(path + '/segmentation/models')
sys.path.append(path + '/segmentation/notebooks')
import layers
import fcn8s
import util
import cityscapes
from colorize import colorize
from class_mean_iou import class_mean_iou

rosbag_name = 'stairs_lappis_2'

pcl_pub = []
TF_L = []
Map_W = 16
Map_B = 16
Map_resolution = 0.08
Max_w = Map_W/Map_resolution
Max_b = Map_B/Map_resolution

Target_frame = 'base_link'
g_zero_mask = []
g_kinect_cloud = []
g_kinect_img_msg = []
g_kinect_img = []
g_cnn_features = []


image_shape = [1, 256, 512, 3]
sess = tf.InteractiveSession()
image_op = tf.placeholder(tf.float32, shape=image_shape)

logits_op = fcn8s.inference(image_op)
predictions_op = layers.predictions(logits_op)
predictions_op_prob = tf.nn.softmax(logits_op)

init_op = tf.global_variables_initializer()
sess.run(init_op)

bridge = CvBridge()

saver = tf.train.Saver()
saver.restore(sess, path + '/tf_models/fcn8s_augment_finetune/' + 'fcn8s_augment.checkpoint-30')
prediction_publisher = rospy.Publisher('/prediction_color', Image, queue_size=1)


def predice_image(img_msg):
    global g_kinect_img

    feature_vision = np.zeros( [1, 256, 512, 34], dtype=np.float32 )
    #np_arr = np.fromstring(img_msg.data, np.uint8)         
    #image = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)    
    image = bridge.imgmsg_to_cv2(img_msg)
    g_kinect_img = image*1
    print('image shape recieved:', image.shape)
    # image = bridge.imgmsg_to_cv2(img_msg)

    image = sp.misc.imresize(image, image_shape[1:], interp='bilinear')

    image = image[..., ::-1] # bgr to rgb

    image = (image - image.mean()) / image.std()
    
    feed_dict = {image_op: image[np.newaxis, ...]}
    
    prediction_label = sess.run(predictions_op, feed_dict=feed_dict)
    feature_vision = sess.run(predictions_op_prob, feed_dict=feed_dict)

#     pickle.dump(prediction_prob, open("/home/xi/workspace/labels/prob.p", "wb"))
    prediction_label = colorize(prediction_label, cityscapes.augmented_labels)
    # image_message = bridge.cv2_to_imgmsg(prediction_label)
    # label_pub.publish(image_message)

    cv2.imshow("prediction_label", prediction_label)
    cv2.waitKey(10)

    # prediction_label = prediction_label[..., ::-1] # rgb to bgr
    # prediction_publisher.publish(bridge.cv2_to_imgmsg(prediction_label))

    print('CNN feature done')
    return feature_vision


def get_transform_matrix(target_frame, source_frame, time_stamp):

    t = TF_L.getLatestCommonTime(target_frame, source_frame)
    translation,rotation = TF_L.lookupTransform(target_frame, source_frame, t)

    # translation,rotation = TF_L.lookupTransform(target_frame, source_frame, time_stamp)
    return TF_L.fromTranslationRotation(translation, rotation)

def transformPointCloud(target_frame, point_cloud):
    r = PointCloud()
    r.header.stamp = point_cloud.header.stamp
    r.header.frame_id = target_frame
    # r.channels = point_cloud.channels

    print('cloud frame', point_cloud.header)

    mat44 = get_transform_matrix(target_frame, point_cloud.header.frame_id, point_cloud.header.stamp) # get transform matrix

    def xf(p):
        xyz = tuple(np.dot(mat44, np.array([p[0], p[1], p[2], 1.0])))[:3]
        return geometry_msgs.msg.Point(*xyz)

    r.points = [xf(p) for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z"), skip_nans=True)]
    return r


def ros_to_pcl(ros_cloud):
    """ Converts a ROS PointCloud2 message to a pcl PointXYZRGB

        Args:
            ros_cloud (PointCloud2): ROS PointCloud2 message

        Returns:
            pcl.PointCloud_PointXYZ: PCL XYZ point cloud
    """
    points_list = []

    mat44 = get_transform_matrix(Target_frame, ros_cloud.header.frame_id, ros_cloud.header.stamp) # get transform matrix

    print(mat44)

    # transformed_cloud = transformPointCloud("base_link", ros_cloud)

    for data in pc2.read_points(ros_cloud, skip_nans=True):
        xyz = tuple(np.dot(mat44, np.array([data[0], data[1], data[2], data[3]])))[:3]
        if abs(data[0]) < Map_W/2-Map_resolution and abs(data[1]) < Map_W/2-Map_resolution:
            if abs(data[0]) > 0.5 or abs(data[1]) > 0.5:
                points_list.append([data[0], data[1], data[2]])

    pcl_data = pcl.PointCloud()
    pcl_data.from_list(points_list)
    # print(pcl_data.to_array())

    return pcl_data 

def show_img(img, name, waitkey, use_mask=True):
    print(name, img.min(), img.max()) 

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

    print(min_img.shape)
    for p in cloud:
        row = int((p[0]+Map_W/2)/Map_resolution)
        col = int((p[1]+Map_W/2)/Map_resolution)
	if row >= Max_w or row < 0 or col >= Max_b or col < 0:
            continue

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

def get_uv_from_xyz(x, y, z):
    fx, fy, cx, cy = 540.68603515625, 540.68603515625, 479.75, 269.75

    u = (fx*x) / z + cx
    v = (fy*y) / z + cy
    return u, v

def get_cloud_uv(base_cloud, mat44, point_row_col):
    # mat44 = get_transform_matrix('kinect2_rgb_optical_frame', cloud_header.frame_id, camera_header.stamp) # get transform matrix

    points_in_cam = []
    points_in_base = []
    uvdh_rc = []
    for p, (row, col)  in zip(base_cloud, point_row_col):
        p_cam = np.dot(mat44, np.array([p[0], p[1], p[2], 1.0]))[:3]
        if p_cam[2] < 1:
            continue
        u, v = get_uv_from_xyz(p_cam[0], p_cam[1], p_cam[2])
        if u < 0 or u >= 960 or v < 0 or v >= 540:
            continue
        points_in_cam.append(p_cam)
        points_in_base.append(p)
        uvdh_rc.append([u, v, p_cam[2], p[2], row, col])

    cloud_in_cam = pcl.PointCloud()
    cloud_in_base = pcl.PointCloud()
    cloud_in_cam.from_list(points_in_cam)
    cloud_in_base.from_list(points_in_base)

    # cloud_header.frame_id = 'kinect2_rgb_optical_frame'
    # publish_cloud(cloud_in_cam, cloud_header, pcl_pub)   

    return uvdh_rc, cloud_in_cam

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


def call_back_cloud(msg):
    global pcl_pub, g_cnn_features
    if g_kinect_cloud == [] or g_cnn_features == []:
        return 0

    g_cnn_features = predice_image(g_kinect_img_msg)

    # label_img = match_img(rosbag_name)
    # if label_img == []:
    #     return 0

    s_time = time.time()
    pcl_cloud = ros_to_pcl(msg)
    pcl_cloud_kinect = ros_to_pcl(g_kinect_cloud)

    # mls = pcl_cloud.make_moving_least_squares()
    # mls.set_search_radius(0.05)
    # cloud_filtered = mls.process()

    # mls = pcl_cloud.make_statistical_outlier_filter()
    # mls.set_mean_k(50)
    # mls.set_std_dev_mul_thresh(1)
    # cloud_filtered = mls.filter()

    cloud_filtered, point_row_col, hdiff_img = filter_cloud_by_minh(pcl_cloud)
    show_img(hdiff_img, 'hdiff_img', 10)
    print('filtered!', time.time()-s_time)

    obs_rgb = detect_narrow_passage(hdiff_img)

    ###############################################
    # compute slope and roughness
    ###############################################
    slope_img, roughness_img = get_slope_roughness_img(cloud_filtered, point_row_col)

    ###############################################
    # build min dist map, start projecting cloud on image
    ###############################################
    mat44 = get_transform_matrix('kinect2_rgb_optical_frame', msg.header.frame_id, g_kinect_cloud.header.stamp) # get transform matrix
    uvd_rc, cloud_in_cam = get_cloud_uv(cloud_filtered, mat44, point_row_col)

    msg.header.frame_id = 'kinect2_rgb_optical_frame'
    trans, cloud_rotated = rotate_map_cloud(cloud_in_cam, pcl_cloud_kinect)
    uvd_rc, cloud_in_cam = get_cloud_uv(cloud_rotated, trans, point_row_col)

    publish_cloud(cloud_rotated, 'kinect2_rgb_optical_frame', pcl_pub)   
    print('published')


    min_dist_img = np.full((540, 960), 0, np.float32)
    for (u, v, dist, h, row, col) in uvd_rc:
        u, v = int(u), int(v)
        dist_pre = min_dist_img[v, u]
        if dist_pre == 0 or dist < dist_pre:
            min_dist_img[v, u] = dist
    ###############################################
    # map all features to img
    ###############################################
    f_hd_img = np.full((540, 960), 0, np.float32)

    features = []
    for (u, v, dist, h, row, col) in uvd_rc:
        u, v = int(u), int(v)
        min_dist = min_dist_img[v, u]
        # if dist > min_dist:
        #     continue 
        
        f_hd_img[v, u] = roughness_img[row, col]

    show_img(f_hd_img, 'f_hd_img', 10, False)
    show_img(min_dist_img, 'min_dist_img', 0, False)


    # print(normal_big[0], normal_small[0])

    # publish_cloud(cloud_filtered, pcl_pub)   
    print('done!', time.time()-s_time)


def call_back_kcloud(msg):
    global g_kinect_cloud

    if g_kinect_cloud == []:
        g_kinect_cloud = msg

def call_back_img(msg):
    global g_cnn_features, g_kinect_img_msg

    if g_cnn_features == []:
        g_cnn_features = 1 #predice_image(msg)
        g_kinect_img_msg = msg

def publish_cloud(cloud, frame_id, publisher):
    #header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id #'base_link_oriented'
    #create pcl from points
    scaled_polygon_pcl = pc2.create_cloud_xyz32(header, cloud.to_array())
    #publish    
    publisher.publish(scaled_polygon_pcl)      

def main(args):
    global pcl_pub, TF_L
    rospy.init_node('pcl_test', anonymous=True)

    TF_L = ros_tf.TransformListener()
    pcl_pub = rospy.Publisher('cloud_test', PointCloud2, queue_size=1)

    rospy.Subscriber("/preprocessed_cloud", PointCloud2, call_back_cloud)
    rospy.Subscriber("/preprocessed_kinect", PointCloud2, call_back_kcloud)
    rospy.Subscriber('/kinect2/qhd/image_color', Image, callback=call_back_img)

    # load_loabls()

    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)
