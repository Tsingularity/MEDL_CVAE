import tensorflow as tf
import tensorflow.contrib.slim as slim
FLAGS = tf.app.flags.FLAGS

class resnet:

	def __init__(self,is_training):
		
		self.input_image = tf.placeholder(dtype=tf.float32,shape=[None,3,128],name='input_image')
		
		self.input_label = tf.placeholder(dtype=tf.float32,shape=[None,100],name='input_label')

		self.input_nlcd = tf.placeholder(dtype=tf.float32,shape=[None,15],name='input_nlcd')

		#logits, end_points = resnet_v2.resnet_v2_50(self.input_image, num_classes=100, is_training=True)

		# flatten_hist = tf.reshape(self.input_image,[-1,96])

		self.keep_prob = tf.placeholder(tf.float32)

		weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)

		# x = tf.expand_dims(self.input_image,-1)

		# #regularizer = slim.l2_regularizer(weight_decay)

		# batch_norm = slim.batch_norm
		# batch_norm_params = {'is_training':is_training,'updates_collections':tf.GraphKeys.UPDATE_OPS,'decay':0.9,'epsilon':0.00001}

		# #Padding: conv2d default is 'SAME'
		# #Padding: pool2d default is 'VALID'
		
		# x = slim.conv2d(scope='conv1',inputs=x,num_outputs=16,kernel_size=[3,3],stride=[3,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = regularizer)

		# x = slim.max_pool2d(scope='pool1',inputs=x,kernel_size=[3,2],stride=[3,2],padding='SAME')

		# x = slim.conv2d(scope='conv2',inputs=x,num_outputs=32,kernel_size=[1,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = regularizer)		

		# x = slim.max_pool2d(scope='pool2',inputs=x,kernel_size=[1,2],stride=[1,2],padding='SAME')

		# x = slim.conv2d(scope='conv3',inputs=x,num_outputs=64,kernel_size=[1,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = regularizer)		

		# x = slim.max_pool2d(scope='pool3',inputs=x,kernel_size=[1,2],stride=[1,2],padding='SAME')

		# x = slim.conv2d(scope='conv4',inputs=x,num_outputs=128,kernel_size=[1,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = regularizer)		

		# x = slim.max_pool2d(scope='pool4',inputs=x,kernel_size=[1,2],stride=[1,2],padding='SAME')

		# x = slim.conv2d(scope='conv5',inputs=x,num_outputs=256,kernel_size=[1,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = regularizer)		

		# x = slim.max_pool2d(scope='pool5',inputs=x,kernel_size=[1,2],stride=[1,2],padding='SAME')

		# x = slim.conv2d(scope='conv6',inputs=x,num_outputs=512,kernel_size=[1,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = regularizer)		

		# x = slim.max_pool2d(scope='pool6',inputs=x,kernel_size=[1,2],stride=[1,2],padding='SAME')

		# x = tf.reshape(x,[-1,512])

		# x = slim.fully_connected(x, 256,weights_regularizer=regularizer, scope='hist/fc_1')
		# x = slim.fully_connected(x, 100,weights_regularizer=regularizer, scope='hist/fc_2')


		# nlcd = slim.fully_connected(self.input_nlcd, 256,weights_regularizer=regularizer,scope='fc/fc_1')
		# nlcd = slim.fully_connected(x, 256,weights_regularizer=regularizer, scope='fc/fc_2')
		# nlcd = slim.fully_connected(x, 100,weights_regularizer=regularizer, scope='fc/fc_3')

		# x = tf.concat([nlcd,x],1)

		# x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)

		# x = slim.fully_connected(inputs=x, num_outputs=100, activation_fn=None, biases_initializer=None, weights_regularizer=regularizer,scope='final_fc')

		flatten_hist = tf.reshape(self.input_image,[-1,3*128])
		self.image_feature_decoder = flatten_hist
		input_x = tf.concat([self.input_nlcd,self.image_feature_decoder],1)
		#x = tf.concat([self.input_nlcd,sample_z],1)

		x = slim.fully_connected(input_x, 512,weights_regularizer=weights_regularizer,scope='decoder/fc_1')
		x = slim.fully_connected(x, 1024,weights_regularizer=weights_regularizer, scope='decoder/fc_2')
		x = slim.fully_connected(x, 499,weights_regularizer=weights_regularizer, scope='decoder/fc_3')

		#x = x+input_x
		
		#dropout
		x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)
		
		self.logits = slim.fully_connected(x, 100, activation_fn=None, weights_regularizer=weights_regularizer,scope='decoder/logits')

		self.output = tf.sigmoid(self.logits,name='decoder/output')

		self.ce_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label,logits=self.logits),1))