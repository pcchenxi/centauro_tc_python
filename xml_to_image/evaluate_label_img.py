import cv2
import numpy as np
import sys


labels 	= np.array([50, 150, 255])
classes = np.array(['flat', 'rough', 'obstacle'])

def main(args):
	while (1):
		files = (raw_input("drag your label and predict img here: "))
		if(files == ''):
			break
		else:
			pathes = files.split() 
			if(len(pathes) != 2):
				print "only accept two files"
				continue
			
			pathes[0] = pathes[0].replace("'", '') 
			pathes[0] = pathes[0].replace(" ", '')
			pathes[1] = pathes[1].replace("'", '') 
			pathes[1] = pathes[1].replace(" ", '')			

			if(pathes[0].find('label')):
				evaluate_img(pathes[0], pathes[1])
			else:
				evaluate_img(pathes[1], pathes[0])

			#cv2.destroyAllWindows()

def get_valid_mask(img_label, img_predict):
	ret1, label_valid    = cv2.threshold(img_label,  0,255,cv2.THRESH_BINARY)
	ret2, predict_valid  = cv2.threshold(img_predict,0,255,cv2.THRESH_BINARY)

	valid_mask = label_valid & predict_valid

	return valid_mask

def repair_img(img):
	rows, cols = img.shape
	for row in range (0, rows):
		for col in range (0, cols):
			if(img[row,col] < 50):
				img[row,col] = 0
			elif(img[row,col] < 150):
				img[row,col] = 50	
			elif(img[row,col] < 255):
				img[row,col] = 150
			else:
				img[row,col] = 255		

	return img

# def get_img_per_class(img_label, img_predict, label):
	# ret1, label_1   = cv2.threshold(img_label, label,  0,cv2.THRESH_TOZERO_INV)
	# ret2, label_2   = cv2.threshold(label_1,   label-1,0,cv2.THRESH_TOZERO)
	# # label_class    = cv2.morphologyEx(label_2, cv2.MORPH_OPEN, kernel)


	# ret2, predict_1 = cv2.threshold(img_predict, label,  0,cv2.THRESH_TOZERO_INV)
	# ret2, predict_2 = cv2.threshold(predict_1, label-1,0,cv2.THRESH_TOZERO)
	# # predict_class    = cv2.morphologyEx(predict_2, cv2.MORPH_OPEN, kernel)

	# valid_mask 		= get_valid_mask(label_2, predict_2)

	# label_class 	= label_2 & valid_mask
	# predict_class   = predict_2 & valid_mask

	# cv2.imshow('label', label_1)
	# cv2.imshow('predict', label_2)
	# cv2.imshow('valid_mask', label_3)


def campare_tp(img_label, img_predict, valid_mask):

	img_diff = abs(img_label - img_predict)
	img_diff = img_diff & valid_mask

	fp = float(cv2.countNonZero(img_diff)) / float (cv2.countNonZero(valid_mask))
	tp = 1-fp
	# print cv2.countNonZero(img_diff), cv2.countNonZero(valid_mask)
	# print 'Overall True Positive: ', tp
	
	# cv2.imshow('valid_mask', valid_mask)
	cv2.imshow('img_diff', img_diff)


def get_class_img(img, label, make_mask):
	ret1, p_1   = cv2.threshold(img, label,  0,cv2.THRESH_TOZERO_INV)
	ret2, p_2   = cv2.threshold(p_1,   label-1,0,cv2.THRESH_TOZERO)

	if(make_mask):
		rect1, p_2 = cv2.threshold(p_2, 0, 255, cv2.THRESH_BINARY)

	return p_2


def campare_predict(img_label, img_predict):

	# np.set_printoptions(precision=2)
	np.set_printoptions(suppress=True)
	matrix = np.zeros( (len(labels), len(labels)), np.float16)

	for i in range(0, len(labels)):
		sub_label 		= get_class_img(img_label, labels[i], 1)
		masked_pridict  = img_predict & sub_label
		# cv2.imshow('sub_label', sub_label)
		# cv2.imshow('masked_pridict', masked_pridict)

		sub_label_count = float(cv2.countNonZero(sub_label))
		print 'for class ', classes[i], ' : ', sub_label_count
		
		for j in range(0, len(labels)):
			if(sub_label_count == 0):
				matrix[i,j] = 0
				continue

			sub_predict = get_class_img(masked_pridict, labels[j], 0)	
			count = float (cv2.countNonZero(sub_predict))
			# print 'predicted as ', classes[j], ' : ', count, ' ', count/sub_label_count

			matrix[i,j] = count/sub_label_count
			# cv2.imshow('sub_predict', sub_predict)
			# cv2.waitKey(0)
	for i in range(0, len(labels)):
		for j in range(0, len(labels)):
			print matrix[i,j]
	print matrix

def compute_one_iou(img_label, img_predict, label):

	tp_count = 0
	fp_count = 0
	fn_count = 0

	rows, cols = img_label.shape
	for row in range (0, rows):
		for col in range (0, cols):

			if(img_predict[row,col] == 0 or img_label[row,col] == 0):
				continue
			if(img_predict[row,col] == label and img_label[row,col] == label):
				tp_count = tp_count + 1
			elif(img_predict[row,col] == label and img_label[row,col] != label):
				fp_count = fp_count + 1
			elif(img_predict[row,col] != label and img_label[row,col] == label):
				fn_count = fn_count + 1

	sum = tp_count + fp_count + fn_count
	iou = float(tp_count) / float(sum)

	print tp_count, fp_count, fn_count, sum, iou

	return iou


def compute_all_ious(img_label, img_predict):

	sum_iou = 0
	for i in range(0, len(labels)):
		print classes[i] + ': '
		iou = compute_one_iou (img_label, img_predict, labels[i])
		sum_iou = sum_iou + iou
	
	mean_iou = sum_iou/len(labels)
	print 'mean_iou: ', mean_iou

def evaluate_img(label_path, predict_path):
	img_label 	= cv2.imread(label_path, 0)
	img_predict = cv2.imread(predict_path, 0)

	# img_predict = repair_img(img_predict)
	
	#### processing ############
	valid_mask  = get_valid_mask(img_label, img_predict)
	img_label   = img_label & valid_mask
	img_predict = img_predict & valid_mask

	# campare_tp(img_label, img_predict, valid_mask)
	# compute_all_ious(img_label, img_predict)
	campare_predict(img_label, img_predict)
	

	# cv2.imshow('img_label', img_label)
	# cv2.imshow('img_predict', img_predict)

	# cv2.imwrite(predict_path, img_predict)
	# cv2.waitKey(0)
	cv2.destroyAllWindows()

	# img 	 = np.zeros((width,height,3), np.uint8)
	# img_grey = np.zeros((width,height,1), np.uint8)



if __name__ == '__main__':
	main(sys.argv)

