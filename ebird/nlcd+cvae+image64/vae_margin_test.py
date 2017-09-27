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
		
		self.input_nlcd = tf.placeholder(dtype=tf.float32,shape=[None,15],name='input_nlcd')
		
		self.input_label = tf.placeholder(dtype=tf.float32,shape=[None,100],name='input_label')

		self.keep_prob = tf.placeholder(tf.float32)

		weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)

		flatten_hist = tf.reshape(self.input_image,[-1,3*64*64])

		# flatten_hist = slim.fully_connected(flatten_hist, 1024,weights_regularizer=weights_regularizer,scope='fig/fc_1')
		# flatten_hist = slim.fully_connected(flatten_hist, 256,weights_regularizer=weights_regularizer, scope='fig/fc_2')
		# flatten_hist = slim.fully_connected(flatten_hist, 25,weights_regularizer=weights_regularizer, scope='fig/fc_3')

		self.image_feature_encoder = flatten_hist
		self.image_feature_decoder = flatten_hist
		
		############## Q(z|X) ###############

		# x = tf.concat([self.input_nlcd,self.image_feature_encoder,self.input_label],1)



		# x = slim.conv2d(scope='encoder/conv1',inputs=self.input_image,num_outputs=32,kernel_size=[5,5],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# x = slim.max_pool2d(scope='encoder/pool1',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		# x = slim.conv2d(scope='encoder/conv2',inputs=x,num_outputs=64,kernel_size=[5,5],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)		

		# x = slim.max_pool2d(scope='encoder/pool2',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		# flatten_hist = tf.reshape(x,[-1,64*7*7])

		# self.image_feature_encoder = flatten_hist

		# x = tf.concat([self.input_nlcd,self.image_feature_encoder,self.input_label],1)

		# x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer,scope='encoder/fc_1')
		# x = slim.fully_connected(x, 1024,weights_regularizer=weights_regularizer, scope='encoder/fc_2')
		# x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='encoder/fc_3')

		# #x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)

		# z_miu = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='encoder/z_miu')
		# z_logvar = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='encoder/z_logvar')

		condition = tf.concat([self.image_feature_encoder,self.input_nlcd],1)

		x = slim.fully_connected(condition, 512,weights_regularizer=weights_regularizer,scope='condition/fc_1')
		x = slim.fully_connected(x, 1024,weights_regularizer=weights_regularizer, scope='condition/fc_2')
		x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='condition/fc_3')
		
		condition_miu = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='condition/z_miu')
		condition_logvar = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='condition/z_logvar')		



		############## Sample_z ###############

		eps = tf.random_normal(shape=tf.shape(condition_miu))
		self.sample_z = condition_miu + tf.exp(condition_logvar / 2) * eps

		############## P(X|z) ###############

		#flatten_hist = tf.reshape(self.input_image,[-1,3*64*64])

		#self.image_feature_decoder = flatten_hist
		x = tf.concat([self.input_nlcd,self.image_feature_decoder,self.sample_z],1)

		x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer,scope='decoder/fc_1')
		x = slim.fully_connected(x, 1024,weights_regularizer=weights_regularizer, scope='decoder/fc_2')
		x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='decoder/fc_3')

		x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)
		
		self.logits = slim.fully_connected(x, 100, activation_fn=None, weights_regularizer=weights_regularizer,scope='decoder/logits')

		self.output = tf.sigmoid(self.logits,name='decoder/output')

		