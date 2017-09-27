import tensorflow as tf
import tensorflow.contrib.slim as slim

FLAGS = tf.app.flags.FLAGS

def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    kld = -0.5 * tf.reduce_sum(1 + (recog_logvar - prior_logvar)
                               - tf.div(tf.pow(prior_mu - recog_mu, 2), tf.exp(prior_logvar))
                               - tf.div(tf.exp(recog_logvar), tf.exp(prior_logvar)), reduction_indices=1)
    return kld


class vae:

	def __init__(self,is_training):

		z_dim = FLAGS.z_dim

		self.input_image = tf.placeholder(dtype=tf.float32,shape=[None,64,64,3],name='input_image')
		
		self.input_label = tf.placeholder(dtype=tf.float32,shape=[None,17],name='input_label')

		self.keep_prob = tf.placeholder(tf.float32)

		weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)

		flatten_hist = tf.reshape(self.input_image,[-1,3*64*64])

		batch_norm = slim.batch_norm
		batch_norm_params = {'is_training':is_training,'updates_collections':tf.GraphKeys.UPDATE_OPS,'decay':0.9,'epsilon':0.00001}

		# self.image_feature_encoder = flatten_hist
		# self.image_feature_decoder = flatten_hist

		# x = slim.fully_connected(flatten_hist, 512,weights_regularizer=weights_regularizer,scope='image/fc_1')
		# #x = slim.fully_connected(x, 1024,weights_regularizer=weights_regularizer, scope='image/fc_2')
		# x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='image/fc_3')
		# x = slim.fully_connected(x, 100,weights_regularizer=weights_regularizer, scope='image/fc_4')
		

		x = slim.conv2d(scope='encoder/image/conv1',inputs=self.input_image,num_outputs=32,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		x = slim.max_pool2d(scope='encoder/image/pool1',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		x = slim.conv2d(scope='encoder/image/conv2',inputs=x,num_outputs=64,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)		

		x = slim.max_pool2d(scope='encoder/image/pool2',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		x = slim.conv2d(scope='encoder/image/conv3',inputs=x,num_outputs=128,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)		

		x = slim.max_pool2d(scope='encoder/image/pool3',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		x = tf.reshape(x,[-1,8*8*128])

		x=slim.fully_connected(x, 256, weights_regularizer=weights_regularizer,scope='encoder/image/fc_1')
		x=slim.fully_connected(x, 100, weights_regularizer=weights_regularizer,scope='encoder/image/fc_2')
		x=slim.fully_connected(x, 50, weights_regularizer=weights_regularizer,scope='image/fc_3')


		self.encoder_image_feature = x
		
		############## Q(z|X) ###############

		# x = tf.concat([self.encoder_image_feature,self.input_label],1)

		# x = slim.fully_connected(x, 50,weights_regularizer=weights_regularizer,scope='encoder/fc_1')
		# #x = slim.fully_connected(x, 100,weights_regularizer=weights_regularizer, scope='encoder/fc_2')
		# #x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='encoder/fc_3')

		# #x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)

		# self.z_miu = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='encoder/z_miu')
		# z_logvar = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='encoder/z_logvar')

		#condition = tf.concat([self.image_feature_encoder,self.input_nlcd],1)
		condition = self.encoder_image_feature

		x = slim.fully_connected(condition, 50,weights_regularizer=weights_regularizer,scope='condition/fc_1')
		# x = slim.fully_connected(x, 100,weights_regularizer=weights_regularizer, scope='condition/fc_2')
		# x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='condition/fc_3')
		
		self.condition_miu = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='condition/z_miu')
		condition_logvar = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='condition/z_logvar')		



		############## Sample_z ###############

		eps = tf.random_normal(shape=tf.shape(self.condition_miu))
		self.sample_z = self.condition_miu + tf.exp(condition_logvar / 2) * eps

		############## P(X|z) ###############

		# x = slim.conv2d(scope='decoder/image/conv1',inputs=self.input_image,num_outputs=32,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# #x = slim.max_pool2d(scope='decoder/image/pool1',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')
		# x = slim.conv2d(scope='decoder/image/pool1',inputs=x,num_outputs=32,kernel_size=[2,2],stride=[2,2],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# x = slim.conv2d(scope='decoder/image/conv2',inputs=x,num_outputs=64,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)		

		# #x = slim.max_pool2d(scope='decoder/image/pool2',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')
		# x = slim.conv2d(scope='decoder/image/pool2',inputs=x,num_outputs=64,kernel_size=[2,2],stride=[2,2],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# x = slim.conv2d(scope='decoder/image/conv3',inputs=x,num_outputs=128,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)		

		# x = slim.max_pool2d(scope='decoder/image/pool3',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		# x = tf.reshape(x,[-1,8*8*128])

		# x=slim.fully_connected(x, 256, weights_regularizer=weights_regularizer,scope='decoder/image/fc_1')
		# x=slim.fully_connected(x, 50, weights_regularizer=weights_regularizer,scope='decoder/image/fc_2')
		#x=slim.fully_connected(x, 100, weights_regularizer=weights_regularizer,scope='image/fc_3')


		self.decoder_image_feature = self.encoder_image_feature

		
		x = tf.concat([self.decoder_image_feature,self.sample_z],1)

		#x = self.image_feature

		x = slim.fully_connected(x, 50,weights_regularizer=weights_regularizer,scope='decoder/fc_1')
		x = slim.fully_connected(x, 50,weights_regularizer=weights_regularizer, scope='decoder/fc_2')
		# x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='decoder/fc_3')

		x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)
		
		self.logits = slim.fully_connected(x, 17, activation_fn=None, weights_regularizer=weights_regularizer,scope='decoder/logits')

		self.output = tf.sigmoid(self.logits,name='decoder/output')