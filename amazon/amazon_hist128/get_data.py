import numpy as np
from PIL import Image

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
		input_labels.append(data[i])

	input_labels = np.array(input_labels) 
	return input_labels