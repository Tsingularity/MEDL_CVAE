import tensorflow as tf
import tensorflow.contrib.slim as slim
FLAGS = tf.app.flags.FLAGS

class fc:

	def __init__(self,is_training):
		
		self.input_image = tf.placeholder(dtype=tf.float32,shape=[None,64,64,3],name='input_image')
		
		self.input_label = tf.placeholder(dtype=tf.float32,shape=[None,17],name='input_label')

		self.keep_prob = tf.placeholder(tf.float32)

		weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)


		batch_norm = slim.batch_norm
		batch_norm_params = {'is_training':is_training,'updates_collections':tf.GraphKeys.UPDATE_OPS,'decay':0.9,'epsilon':0.00001}
		

		#flatten_hist = tf.reshape(self.input_image,[-1,3*64*64])

		# x = slim.fully_connected(flatten_hist, 512,weights_regularizer=weights_regularizer,scope='decoder/fc_1')
		# x = slim.fully_connected(x, 1024,weights_regularizer=weights_regularizer, scope='decoder/fc_2')
		# x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='decoder/fc_3')

		all_logits = []
		all_output = []

		x = slim.conv2d(scope='decoder/conv1',inputs=self.input_image,num_outputs=32,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		x = slim.max_pool2d(scope='decoder/pool1',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		x = slim.conv2d(scope='decoder/conv2',inputs=x,num_outputs=64,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)		

		x = slim.max_pool2d(scope='decoder/pool2',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		x = slim.conv2d(scope='decoder/conv3',inputs=x,num_outputs=128,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)		

		x = slim.max_pool2d(scope='decoder/pool3',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		feature = tf.reshape(x,[-1,8*8*128])

		for i in range(17):
			if i == 0 :
				current_input_x = feature
			else:
				current_output = tf.concat(all_output,1)
				current_input_x = tf.concat([feature,current_output],1)

			x = slim.fully_connected(current_input_x, 512,weights_regularizer=weights_regularizer)
			x = slim.fully_connected(x, 1024,weights_regularizer=weights_regularizer)
			x = slim.fully_connected(x, 17,weights_regularizer=weights_regularizer)

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