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


		flatten_hist = tf.reshape(self.input_image,[-1,3*64*64])
		flatten_hist = tf.concat([flatten_hist,self.input_nlcd],1)
		x = slim.fully_connected(flatten_hist, 512,weights_regularizer=weights_regularizer,scope='decoder/fc_1')
		x = slim.fully_connected(x, 1024,weights_regularizer=weights_regularizer, scope='decoder/fc_2')
		flatten_hist = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='decoder/fc_3')

		all_logits = []
		all_output = []

		for i in range(100):
			if i == 0 :
				current_input_x = flatten_hist
			else:
				current_output = tf.concat(all_output,1)
				current_input_x = tf.concat([flatten_hist,current_output],1)

			x = slim.fully_connected(current_input_x, 256,weights_regularizer=weights_regularizer)
			x = slim.fully_connected(x, 100,weights_regularizer=weights_regularizer)
			#x = slim.fully_connected(x, 17,weights_regularizer=weights_regularizer)

			x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)
			all_logits.append(slim.fully_connected(x, 1, activation_fn=None, weights_regularizer=weights_regularizer))
			all_output.append(tf.sigmoid(all_logits[i]))

		final_logits = tf.concat(all_logits,1)
		final_output = tf.sigmoid(final_logits)

		self.output = final_output
		self.ce_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label,logits=final_logits),1))

		slim.losses.add_loss(self.ce_loss)
		tf.summary.scalar('ce_loss',self.ce_loss)
		
		# l2 loss
		self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())
		tf.summary.scalar('l2_loss',self.l2_loss)

		#total loss
		self.total_loss = slim.losses.get_total_loss()
		tf.summary.scalar('total_loss',self.total_loss)

		#self.output = tf.sigmoid(x)