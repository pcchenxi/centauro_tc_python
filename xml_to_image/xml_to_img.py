import xml.etree.ElementTree as ET
import cv2, os
import numpy as np
import sys

## define class and color
flat 			= np.array([255,50,50])
obstacle 		= np.array([50,50,255])
rough 			= np.array([50,255,255])

## define grey class label
flat_label 		= 50
rough_label		= 100
obstacle_label 	= 150
stair_label 	= 200

def main(args):
	# while (1):
	# 	xml_path = (raw_input("drag your xml file here or click enter to exit: "))
	# 	print xml_path
	# 	if(xml_path == ''):
	# 		break
	# 	else:
	# 		xml_path = xml_path.replace("'", '') 
	# 		xml_path = xml_path.replace(" ", '')
	# 		print xml_path
	# 		process_xml(xml_path)
	# 		# cv2.destroyAllWindows()

	folder_path = '/home/xi/centauro_img/stairs/'
	file_list = os.listdir(folder_path)
	count = 1
	for file_name in file_list:
		if file_name.find('.xml') != -1:
			process_xml(folder_path + file_name)
			count += 1

	print('feature loaded:', count)

def process_xml(xml_path):

	tree = ET.parse(xml_path)
	root = tree.getroot()

	## save the image with the same name of the origian file
	rgb_path = xml_path[:-4] + '.jpg'
	file_path_color = xml_path.replace(".xml", "_label_color.png")
	file_path_grey 	= xml_path.replace(".xml", "_label.png")

	print('path',file_path_grey)
	#img = cv2.imread('./image/1473440069.193839412_img.jpg')
	#cv2.imshow('img', img)
	#cv2.waitKey(10)


	## get image size and init black image
	rgb_img = cv2.imread(rgb_path, 1)
	height 	 = rgb_img.shape[1] #int(root[3][1].text)
	width 	 = rgb_img.shape[0] #int(root[3][0].text)

	img 	 = np.zeros((width,height,3), np.uint8)
	img_grey = np.zeros((width,height,1), np.uint8)

	img_grey = draw_one_class(root, img_grey, 'safe', flat_label)
	img_grey = draw_one_class(root, img_grey, 'rough', rough_label)
	img_grey = draw_one_class(root, img_grey, 'stair', stair_label)
	img_grey = draw_one_class(root, img_grey, 'obstacle', obstacle_label)

	# cv2.imshow('img', img)
	# cv2.moveWindow('img', 0, 0)
	# cv2.imwrite(file_path_color, img)
        # print(file_path_grey)
	cv2.imwrite(file_path_grey, img_grey)
	cv2.imshow('img', img_grey)
	cv2.imshow('rgb_img', rgb_img)
	cv2.waitKey(0)
	# cv2.waitKey(100)

def draw_one_class(root, img_grey, type_name, color):
	for object in root.iter('object'):
		class_type = object[0].text
		delete = int(object[1].text)
		if delete:
			continue
		if(class_type == type_name):
			# color = obstacle
			label = color
		# if (class_type == "rough"):
		# 	color = rough
		# 	label = rough_label
		# if (class_type == "flat"):
		# 	color = flat
		# 	label = flat_label
		else:
			continue

		for polygon in object.iter('polygon'):
			print(class_type)
			point_num = len(polygon.findall('pt'))

			p = np.zeros((point_num, 2))
			index = 0
			for pt in object.iter('pt'):
				p[index][0] = pt[0].text
				p[index][1] = pt[1].text
				index += 1

			# cv2.fillPoly(img, np.array([p], np.int32), color)
			cv2.fillPoly(img_grey, np.array([p], np.int32), label, 4)
			# cv2.imshow('img', img_grey)
			# cv2.waitKey(0)
	return img_grey


if __name__ == '__main__':
	main(sys.argv)

