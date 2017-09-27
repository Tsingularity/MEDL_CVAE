import numpy as np
from PIL import Image

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def get_jpg_train(images,my_order):

	input_images = []

	for i in my_order:
		input_images.append(images[i])

	input_images = np.array(input_images)
	return input_images

def get_jpg_test(images,my_order):

	return get_jpg_train(images,my_order)

def get_label(data,my_order):

	input_labels = []
	
	for i in my_order:
		input_labels.append(data[i][2:102])

	input_labels = np.array(input_labels) 
	return input_labels

def get_nlcd(data,my_order):

	input_labels = []
	
	for i in my_order:
		input_labels.append(data[i][102:])

	input_labels = np.array(input_labels) 
	return input_labels