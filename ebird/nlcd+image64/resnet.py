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


		batch_norm = slim.batch_norm
		batch_norm_params = {'is_training':is_training,'updates_collections':tf.GraphKeys.UPDATE_OPS,'decay':0.9,'epsilon':0.00001}


		flatten_hist = tf.reshape(self.input_image,[-1,3*64])
		

		flatten_hist = tf.reshape(self.input_image,[-1,3*64*64])
		self.image_feature_decoder = flatten_hist
		input_x = tf.concat([self.input_nlcd,self.image_feature_decoder],1)
		#x = tf.concat([self.input_nlcd,sample_z],1)

		x = slim.fully_connected(input_x, 512,weights_regularizer=weights_regularizer,scope='decoder/fc_1')
		x = slim.fully_connected(x, 1024,weights_regularizer=weights_regularizer, scope='decoder/fc_2')
		x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='decoder/fc_3')

		#x = x+input_x

		# x = slim.conv2d(scope='decoder/conv1',inputs=self.input_image,num_outputs=32,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# x = slim.max_pool2d(scope='decoder/pool1',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		# x = slim.conv2d(scope='decoder/conv2',inputs=x,num_outputs=64,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)		

		# x = slim.max_pool2d(scope='decoder/pool2',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		# x = slim.conv2d(scope='decoder/conv3',inputs=x,num_outputs=128,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)		

		# x = slim.max_pool2d(scope='decoder/pool3',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		# x = tf.reshape(x,[-1,8*8*128])

		# x = tf.concat([self.input_nlcd,x],1)

		# x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer,scope='encoder/fc_1')
		# x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer, scope='encoder/fc_2')
		
		#dropout
		x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)
		
		self.logits = slim.fully_connected(x, 100, activation_fn=None, weights_regularizer=weights_regularizer,scope='decoder/logits')

		self.output = tf.sigmoid(self.logits,name='decoder/output')

		self.ce_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label,logits=self.logits),1))

		tf.summary.scalar('ce_loss',self.ce_loss)

		slim.losses.add_loss(self.ce_loss)		

		self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())

		tf.summary.scalar('l2_loss',self.l2_loss)

		self.total_loss = slim.losses.get_total_loss()

		tf.summary.scalar('total_loss',self.total_loss)

		#self.output = tf.sigmoid(x)