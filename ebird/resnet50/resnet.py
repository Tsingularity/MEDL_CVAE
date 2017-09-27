import tensorflow as tf
import tensorflow.contrib.slim as slim
import resnet_v2

class resnet:

	def __init__(self,is_training):
		
		self.input_image = tf.placeholder(dtype=tf.float32,shape=[None,224,224,3],name='input_image')
		
		self.input_label = tf.placeholder(dtype=tf.float32,shape=[None,100],name='input_label')

		logits, end_points = resnet_v2.resnet_v2_50(self.input_image, num_classes=100, is_training=True)

		self.ce_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label,logits=logits),1))

		tf.summary.scalar('ce_loss',self.ce_loss)

		slim.losses.add_loss(self.ce_loss)		

		self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())

		tf.summary.scalar('l2_loss',self.l2_loss)

		self.total_loss = slim.losses.get_total_loss()

		tf.summary.scalar('total_loss',self.total_loss)

		self.output = tf.sigmoid(logits)