# for cnn
from sensor_msgs.msg import Image
import sys, cv2
import tensorflow as tf
import scipy as sp
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

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
# prediction_publisher = rospy.Publisher('/prediction_color', Image, queue_size=1)


def predice_image(img_msg):
    global g_kinect_img

    feature_vision = np.zeros( [1, 256, 512, 34], dtype=np.float32 )
    #np_arr = np.fromstring(img_msg.data, np.uint8)         
    #image = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)    
    image = bridge.imgmsg_to_cv2(img_msg)
    g_kinect_img = image*1
    print('     image shape recieved:', image.shape)
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

    print(' CNN feature done')
    return feature_vision