import numpy as np
from PIL import Image

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def get_jpg_train(images,my_order):

	input_images = []

	for i in my_order:
		img = Image.open('../real_pic/'+images[i])
		img = img.resize((224,224),Image.BILINEAR)
		flip = np.random.randint(2)==1
		if flip:
			img = img.transpose(Image.FLIP_LEFT_RIGHT)

		angle = np.random.randint(4)*90.0
		img = img.rotate(angle,Image.BILINEAR)
		single_image = np.array(img)
		input_images.append(single_image)

	input_images = np.array(input_images)
	input_images = np.split(input_images,3,axis=3)
	means = [_R_MEAN,_G_MEAN,_B_MEAN]
	for j in range(3):
		input_images[j] = input_images[j]-means[j]
	input_images = np.concatenate(input_images,3)
	return input_images

def get_jpg_test(images,my_order):

	input_images = []

	for i in my_order:
		img = Image.open('../real_pic/'+images[i])
		img = img.resize((224,224),Image.BILINEAR)
		single_image = np.array(img)
		input_images.append(single_image)

	input_images = np.array(input_images)
	input_images = np.split(input_images,3,axis=3)
	means = [_R_MEAN,_G_MEAN,_B_MEAN]
	for j in range(3):
		input_images[j] = input_images[j]-means[j]

	input_images = np.concatenate(input_images,3)

	return input_images

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