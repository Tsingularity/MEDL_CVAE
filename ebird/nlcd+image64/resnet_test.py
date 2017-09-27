import tensorflow as tf
import tensorflow.contrib.slim as slim
FLAGS = tf.app.flags.FLAGS

class resnet:

	def __init__(self,is_training):
		
		self.input_image = tf.placeholder(dtype=tf.float32,shape=[None,64,64,3],name='input_image')
		
		self.input_label = tf.placeholder(dtype=tf.float32,shape=[None,100],name='input_label')

		self.input_nlcd = tf.placeholder(dtype=tf.float32,shape=[None,15],name='input_nlcd')

		#logits, end_points = resnet_v2.resnet_v2_50(self.input_image, num_classes=100, is_training=True)

		# flatten_hist = tf.reshape(self.input_image,[-1,96])

		self.keep_prob = tf.placeholder(tf.float32)

		weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)


		#flatten_hist = tf.reshape(self.input_image,[-1,3*64])
		flatten_hist = tf.reshape(self.input_image,[-1,3*64*64])
		self.image_feature_decoder = flatten_hist
		input_x = tf.concat([self.input_nlcd,self.image_feature_decoder],1)
		#x = tf.concat([self.input_nlcd,sample_z],1)

		x = slim.fully_connected(input_x, 512,weights_regularizer=weights_regularizer,scope='decoder/fc_1')
		x = slim.fully_connected(x, 1024,weights_regularizer=weights_regularizer, scope='decoder/fc_2')
		x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='decoder/fc_3')

		#x = x+input_x
		
		#dropout
		x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)
		
		self.logits = slim.fully_connected(x, 100, activation_fn=None, weights_regularizer=weights_regularizer,scope='decoder/logits')

		self.output = tf.sigmoid(self.logits,name='decoder/output')

		self.ce_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label,logits=self.logits),1))

		#self.output = tf.sigmoid(x)