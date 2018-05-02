import rosbag
import numpy as np
import tf, rospy
import geometry_msgs.msg
import pcl
import sensor_msgs.point_cloud2 as pc2
from pcl import IterativeClosestPoint, GeneralizedIterativeClosestPoint, IterativeClosestPointNonLinear

trans_to_kinect = []

def extract_msgs(rosbag_name, listener, broadcaster):
    global trans_to_kinect
    msg_ready = False

    sourse_bag = rosbag.Bag(rosbag_name)
    msg_map = []
    msg_img = []
    msg_kcloud = []

    listener.clear()
    for topic, msg, t in sourse_bag.read_messages():
        if topic == '/tf':
            broadcaster.sendTransformMessage(msg.transforms[0])
        if topic == '/mapping_nodelet/pointcloud' and msg_map == []:
            msg_map = msg 
            msg_ready = True 
            # print('map cloud')
        if topic == '/kinect2/qhd/image_color' and msg_img == [] and msg_ready:
            msg_img = msg 
            # print('img')
        if topic == '/kinect2/sd/points' and msg_kcloud == [] and msg_ready:
            msg_kcloud = msg
            # print('kcloud')
    
    print('     transform set', msg_map.header.frame_id)
    translation,rotation = listener.lookupTransform('kinect2_rgb_optical_frame', msg_map.header.frame_id, rospy.Time(0))
    trans_to_kinect = listener.fromTranslationRotation(translation, rotation)

    # print(trans_to_kinect)
    return msg_map, msg_img, msg_kcloud

def pre_process_cloud(map_cloud_msg, k_ckoud_msg, max_x, max_y, resolution):
    points_list = []
    for data in pc2.read_points(map_cloud_msg, skip_nans=True):
        if abs(data[0]) < max_x-resolution and abs(data[1]) < max_y-resolution:
        # if (data[0]) > 0 and data[0] < 5 and abs(data[1]) < 4-resolution and data[2] < 4:
            if abs(data[0]) > 0.5 or abs(data[1]) > 0.5:
                points_list.append([data[0], data[1], data[2]])

    pcl_data = pcl.PointCloud()
    pcl_data.from_list(points_list)

    fil = pcl_data.make_ApproximateVoxelGrid()
    fil.set_leaf_size(resolution, resolution, resolution)
    result = fil.filter()

    points_list_k = []
    for data in pc2.read_points(k_ckoud_msg, skip_nans=True):
        if data[2] > 5 or data[2] < 2.5 or abs(data[0]) > 1.5:   #################################
            continue
        points_list_k.append([data[0], data[1], data[2]])

    pcl_k_data = pcl.PointCloud()
    pcl_k_data.from_list(points_list_k)

    return result, pcl_k_data


rgb_imgs, label_imgs, img_path = [], [], []

def get_uv_from_xyz(x, y, z):
    fx, fy, cx, cy = 540.68603515625, 540.68603515625, 479.75, 269.75

    u = (fx*x) / z + cx
    v = (fy*y) / z + cy
    return u, v

def trans_to_kinect_frame(map_cloud):
    # transform cloud to image frame
    point_image = []
    for p in map_cloud:
        if p[0] > 2.5 and p[0] < 5 and abs(p[1])<1.5 and p[2] < 5:      #######################################
            # print(trans_to_kinect)
            p_k = np.dot(trans_to_kinect, np.array([p[0], p[1], p[2], 1.0]))[:3]
            u, v = get_uv_from_xyz(p_k[0], p_k[1], p_k[2])
            if u < 0 or u >= 960 or v < 0 or v >= 540:
                continue
            point_image.append(p_k)

    cloud_rotated = pcl.PointCloud()
    cloud_rotated.from_list(point_image)    

    return cloud_rotated

def rotate_map_cloud(map_cloud, kinect_cloud):
    print('     start icp')

    transf = []
    map_trans = trans_to_kinect_frame(map_cloud)
    
    icp = map_cloud.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(map_trans, kinect_cloud, max_iter=1000)

    point_rotated = []
    for p in map_cloud:
        p_r = np.dot(trans_to_kinect, np.array([p[0], p[1], p[2], 1.0]))[:3]
        p_r = np.dot(transf, np.array([p_r[0], p_r[1], p_r[2], 1.0]))[:3]

        point_rotated.append(p_r)

    cloud_rotated = pcl.PointCloud()
    cloud_rotated.from_list(point_rotated)

    print('     done icp')
    return cloud_rotated


# rospy.init_node('msgs_extractor')
# listener = tf.TransformListener()
# broadcaster = tf.TransformBroadcaster()
# bag_name = '/home/xi/data_recorded/A_1.bag'

# msg_map, msg_img, msg_kcloud = extract_msgs(bag_name)
# pre_process_cloud(msg_map, 4, 4, 0.05)


    # with rosbag.Bag('/home/xi/data_recorded/A_croped.bag', 'w') as outbag:
        # for topic, msg, t in sourse_bag.read_messages(topics=['/tf']):
    # topics = sourse_bag.get_type_and_topic_info()[1].keys()
    # topics_dict = {}
    # for i in range(len(topics)):
    #     topics_dict[topics[i]] = i

    # print(topics_dict)


    # a = []
    # msg_buffer = np.full(len(topics), a)
    # print(msg_buffer)

    # types = []
    # for i in range(0,len(sourse_bag.get_type_and_topic_info()[1].values())):
    #     types.append(sourse_bag.get_type_and_topic_info()[1].values()[i][0])

    # for topic, msg, t in sourse_bag.read_messages():
    #     print(topic, msg)
    #     break
    #     if num_msgs < 1:
    #         break
    #     num_msgs -= 1
        # outbag.write(topic, msg, t)
