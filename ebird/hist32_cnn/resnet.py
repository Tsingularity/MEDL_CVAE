import tensorflow as tf
import tensorflow.contrib.slim as slim
FLAGS = tf.app.flags.FLAGS

class resnet:

	def __init__(self,is_training):
		
		self.input_image = tf.placeholder(dtype=tf.float32,shape=[None,3,32],name='input_image')
		
		self.input_label = tf.placeholder(dtype=tf.float32,shape=[None,100],name='input_label')

		#logits, end_points = resnet_v2.resnet_v2_50(self.input_image, num_classes=100, is_training=True)

		# flatten_hist = tf.reshape(self.input_image,[-1,96])

		self.keep_prob = tf.placeholder(tf.float32)

		regularizer=slim.l2_regularizer(FLAGS.weight_decay)

		x = tf.expand_dims(self.input_image,-1)

		#regularizer = slim.l2_regularizer(weight_decay)

		batch_norm = slim.batch_norm
		batch_norm_params = {'is_training':is_training,'updates_collections':tf.GraphKeys.UPDATE_OPS,'decay':0.9,'epsilon':0.00001}

		#Padding: conv2d default is 'SAME'
		#Padding: pool2d default is 'VALID'
		
		x = slim.conv2d(scope='conv1',inputs=x,num_outputs=16,kernel_size=[3,3],stride=[3,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = regularizer)

		x = slim.max_pool2d(scope='pool1',inputs=x,kernel_size=[3,2],stride=[3,2],padding='SAME')

		x = slim.conv2d(scope='conv2',inputs=x,num_outputs=32,kernel_size=[1,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = regularizer)		

		x = slim.max_pool2d(scope='pool2',inputs=x,kernel_size=[1,2],stride=[1,2],padding='SAME')

		x = slim.conv2d(scope='conv3',inputs=x,num_outputs=64,kernel_size=[1,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = regularizer)		

		x = slim.max_pool2d(scope='pool3',inputs=x,kernel_size=[1,2],stride=[1,2],padding='SAME')

		x = slim.conv2d(scope='conv4',inputs=x,num_outputs=128,kernel_size=[1,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = regularizer)		

		x = slim.max_pool2d(scope='pool4',inputs=x,kernel_size=[1,2],stride=[1,2],padding='SAME')

		x = slim.conv2d(scope='conv5',inputs=x,num_outputs=256,kernel_size=[1,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = regularizer)		

		x = slim.max_pool2d(scope='pool5',inputs=x,kernel_size=[1,2],stride=[1,2],padding='SAME')

		x = tf.reshape(x,[-1,256])

		x = slim.fully_connected(x, 256,weights_regularizer=regularizer, scope='fc/fc_1')
		x = slim.fully_connected(x, 100,weights_regularizer=regularizer, scope='fc/fc_2')

		x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)

		x = slim.fully_connected(inputs=x, num_outputs=100, activation_fn=None, biases_initializer=None, weights_regularizer=regularizer,scope='fc/fc_4')

		self.ce_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label,logits=x),1))

		tf.summary.scalar('ce_loss',self.ce_loss)

		slim.losses.add_loss(self.ce_loss)		

		self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())

		tf.summary.scalar('l2_loss',self.l2_loss)

		self.total_loss = slim.losses.get_total_loss()

		tf.summary.scalar('total_loss',self.total_loss)

		self.output = tf.sigmoid(x)