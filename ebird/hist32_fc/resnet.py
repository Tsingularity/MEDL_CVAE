import tensorflow as tf
import tensorflow.contrib.slim as slim
FLAGS = tf.app.flags.FLAGS

class resnet:

	def __init__(self,is_training):
		
		self.input_image = tf.placeholder(dtype=tf.float32,shape=[None,3,32],name='input_image')
		
		self.input_label = tf.placeholder(dtype=tf.float32,shape=[None,100],name='input_label')

		#logits, end_points = resnet_v2.resnet_v2_50(self.input_image, num_classes=100, is_training=True)

		flatten_hist = tf.reshape(self.input_image,[-1,96])

		self.keep_prob = tf.placeholder(tf.float32)

		weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)

		x = slim.fully_connected(flatten_hist, 256,weights_regularizer=weights_regularizer,scope='fc/fc_1')
		x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer, scope='fc/fc_2')
		x = slim.fully_connected(x, 100,weights_regularizer=weights_regularizer, scope='fc/fc_3')

		x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)

		x = slim.fully_connected(inputs=x, num_outputs=100, activation_fn=None, biases_initializer=None, weights_regularizer=weights_regularizer,scope='fc/fc_4')

		self.ce_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label,logits=x),1))

		tf.summary.scalar('ce_loss',self.ce_loss)

		slim.losses.add_loss(self.ce_loss)		

		self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())

		tf.summary.scalar('l2_loss',self.l2_loss)

		self.total_loss = slim.losses.get_total_loss()

		tf.summary.scalar('total_loss',self.total_loss)

		self.output = tf.sigmoid(x)