#!/usr/bin/env python
###########################################################################
## you can change the rate in "rate = rospy.Rate(1)"
##
## don't forget to run "roscore" first
##
##
## the command to run rosbag is "rosbag play grass_1.bag"
##
###########################################################################


# ros images
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2

import rospy


import sys, os, os.path, numpy as np
import time
import cv2

saved = False

bridge = CvBridge()

def callback(data, num_images):
    	global saved
	try:
		if saved:
		    return
		cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
		time_n = time.time()
		cv2.imwrite('/home/xi/centauro_imgs/' + str(time_n) + '.jpg', cv_image)
		#cv2.imshow("Image ori", cv_image)
		cv2.waitKey(10)
		print "call back"
		saved = True
	except CvBridgeError as e:
		print(e)
	

def main(args):
	global cv_image
	rospy.init_node('caffe_test', anonymous=True)
	# image_sub = rospy.Subscriber("/kinect2/qhd/image_color", Image, callback)

	# rospy.spin()
	rate = rospy.Rate(0.2)   ## 2 means read image for every 0.5 second
        num_images = 0
	while not rospy.is_shutdown():
		msg = rospy.wait_for_message("/kinect2/qhd/image_color", Image)  # change the rostopic name here
		callback(msg, num_images)
		num_images = num_images + 1
		print('images processed', num_images)
		rate.sleep()
		# rospy.spinOnce()
		

	# cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)

