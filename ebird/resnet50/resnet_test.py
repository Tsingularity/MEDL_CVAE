import tensorflow as tf
import tensorflow.contrib.slim as slim
import resnet_v2

class resnet:

	def __init__(self,is_training):
		
		self.input_image = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3],name='input_image')

		logits, end_points = resnet_v2.resnet_v2_50(self.input_image, num_classes=100, is_training=True)

		self.feature = logits

		self.output = tf.sigmoid(logits)