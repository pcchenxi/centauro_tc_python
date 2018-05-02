import ros, pcl, sys, os
import tf as ros_tf
from sensor_msgs.msg import PointCloud, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from tool_functions.extract_msg import *
from tool_functions.geometric_feature_extractor import *
from tool_functions.cnn_feature_extractor import *
from tool_functions.feature_fuser import *

Map_W = 32
Map_B = 32
Map_resolution = 0.08
Map_r = Map_W/Map_resolution
Map_c = Map_B/Map_resolution

def publish_cloud(cloud, frame_id, publisher):
    #header
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id #'base_link_oriented'
    #create pcl from points
    scaled_polygon_pcl = pc2.create_cloud_xyz32(header, cloud.to_array())
    #publish    
    publisher.publish(scaled_polygon_pcl)     

def publish_msg(msg, frame_id, publisher):
    #header
    msg.header.frame_id = frame_id
    msg.header.stamp = rospy.Time.now()
    publisher.publish(msg)   

def get_cloud_uv(base_cloud, point_row_col):
    # mat44 = get_transform_matrix('kinect2_rgb_optical_frame', cloud_header.frame_id, camera_header.stamp) # get transform matrix

    points_in_cam = []
    uvd_rc = []
    for p, (row, col)  in zip(base_cloud, point_row_col):
        if p[2] < 1:
            continue
        u, v = get_uv_from_xyz(p[0], p[1], p[2])
        if u < 0 or u >= 960 or v < 0 or v >= 540:
            continue

        points_in_cam.append(p)
        uvd_rc.append([u, v, p[2], row, col])

    cloud_in_cam = pcl.PointCloud()
    cloud_in_cam.from_list(points_in_cam)

    return uvd_rc, cloud_in_cam

def main(args):
    rospy.init_node('feature_extractor', anonymous=True)

    # TF_L = ros_tf.TransformListener()
    pcl_pub = rospy.Publisher('cloud_test', PointCloud2, queue_size=1)
    pcl_pub_k = rospy.Publisher('cloud_test_k', PointCloud2, queue_size=1)

    # rospy.Subscriber("/preprocessed_cloud", PointCloud2, call_back_cloud)
    # rospy.Subscriber("/preprocessed_kinect", PointCloud2, call_back_kcloud)
    # rospy.Subscriber('/kinect2/qhd/image_color', Image, callback=call_back_img)

    listener = ros_tf.TransformListener()
    broadcaster = ros_tf.TransformBroadcaster()

    rosbag_list = os.listdir('/home/xi/data_recorded/testing_data')
    print(len(rosbag_list))
    # bag_name = '/home/xi/data_recorded/museum/stairs_lappis_6.bag'

    for i in range(len(rosbag_list)):
        bag_name = rosbag_list[i]
        try:
            found_match = load_loabls('/home/xi/data_recorded/testing_data/'+bag_name)
        except:
            continue 

        if not found_match:
            continue

        print('processing rosbag:', i, bag_name)

        # extract messages
        msg_map, msg_img, msg_kcloud = extract_msgs('/home/xi/data_recorded/testing_data/'+bag_name, listener, broadcaster)

        # cloud preprocessing
        pcl_cloud, kinect_cloud = pre_process_cloud(msg_map, msg_kcloud, Map_W/2, Map_B/2, Map_resolution)

        cloud_kinect = rotate_map_cloud(pcl_cloud, kinect_cloud)
        # publish_cloud(cloud_kinect, 'kinect2_rgb_optical_frame', pcl_pub)   

        hdiff_img, slope_img, roughness_img, point_row_col = compute_geometric_features(pcl_cloud, Map_W, Map_B, Map_resolution)
        feature_vision = predice_image(msg_img)

        # get point uv
        uvd_rc, cloud_in_cam = get_cloud_uv(cloud_kinect, point_row_col)
        publish_cloud(cloud_in_cam, 'kinect2_rgb_optical_frame', pcl_pub)   
        publish_msg(msg_kcloud, 'kinect2_rgb_optical_frame', pcl_pub_k)   

        save_all_features(hdiff_img, slope_img, roughness_img, uvd_rc, feature_vision)

    print('finish all rosbags')
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)