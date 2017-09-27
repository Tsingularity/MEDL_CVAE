import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

class fc:

	def __init__(self,is_training):
		
		self.input_nlcd = tf.placeholder(dtype=tf.float32,shape=[None,15],name='input_image')
		
		self.input_label = tf.placeholder(dtype=tf.float32,shape=[None,100],name='input_label')

		self.keep_prob = tf.placeholder(tf.float32)

		weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)

		x = slim.fully_connected(self.input_nlcd, 256,weights_regularizer=weights_regularizer,scope='fc/fc_1')
		x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer, scope='fc/fc_2')
		x = slim.fully_connected(x, 100,weights_regularizer=weights_regularizer, scope='fc/fc_3')

		x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)

		x = slim.fully_connected(inputs=x, num_outputs=100, activation_fn=None, biases_initializer=None, weights_regularizer=weights_regularizer,scope='fc/fc_4')

		self.output = tf.sigmoid(x)