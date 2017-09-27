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

		batch_norm = slim.batch_norm
		batch_norm_params = {'is_training':is_training,'updates_collections':tf.GraphKeys.UPDATE_OPS,'decay':0.9,'epsilon':0.00001}
		
		############## Q(z|X) ###############


		# conv1 = slim.conv2d(scope='encoder/conv1',inputs=self.input_image,num_outputs=16,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# conv2 = slim.conv2d(scope='encoder/conv2',inputs=conv1,num_outputs=16,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# pool1 = slim.max_pool2d(scope='encoder/pool1',inputs=tf.concat([self.input_image,conv2],3),kernel_size=[2,2],stride=[2,2],padding='SAME')

		# conv3 = slim.conv2d(scope='encoder/conv3',inputs=pool1,num_outputs=32,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# conv4 = slim.conv2d(scope='encoder/conv4',inputs=conv3,num_outputs=32,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# pool2 = slim.max_pool2d(scope='encoder/pool2',inputs=tf.concat([pool1,conv4],3),kernel_size=[2,2],stride=[2,2],padding='SAME')

		# conv5 = slim.conv2d(scope='encoder/conv5',inputs=pool2,num_outputs=64,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# conv6 = slim.conv2d(scope='encoder/conv6',inputs=conv5,num_outputs=64,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# pool3 = slim.max_pool2d(scope='encoder/pool3',inputs=tf.concat([pool2,conv6],3),kernel_size=[2,2],stride=[2,2],padding='SAME')

		# conv7 = slim.conv2d(scope='encoder/conv7',inputs=pool3,num_outputs=128,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# conv8 = slim.conv2d(scope='encoder/conv8',inputs=conv7,num_outputs=128,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# pool4 = slim.max_pool2d(scope='encoder/pool4',inputs=tf.concat([pool3,conv8],3),kernel_size=[2,2],stride=[2,2],padding='SAME')

		# conv9 = slim.conv2d(scope='encoder/conv9',inputs=pool4,num_outputs=256,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# conv10 = slim.conv2d(scope='encoder/conv10',inputs=conv9,num_outputs=256,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# pool5 = slim.max_pool2d(scope='encoder/pool5',inputs=tf.concat([pool4,conv10],3),kernel_size=[4,4],stride=[4,4],padding='SAME')

		# conv11 = slim.conv2d(scope='encoder/conv11',inputs=pool5,num_outputs=25,kernel_size=[1,1],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# flatten_hist = tf.reshape(conv11,[-1,25])

		# flatten_hist = slim.fully_connected(flatten_hist, 128,weights_regularizer=weights_regularizer,scope='encoder/flatten_hist_1')

		# flatten_hist = slim.fully_connected(flatten_hist, 25,weights_regularizer=weights_regularizer,scope='encoder/flatten_hist_2')

		# flatten_hist = tf.reshape(self.input_image,[-1,3*28*28])

		#self.image_feature_encoder = flatten_hist

		#self.image_feature_decoder = tf.stop_gradient(flatten_hist)


		x = slim.conv2d(scope='decoder/conv1',inputs=self.input_image,num_outputs=32,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		x = slim.max_pool2d(scope='decoder/pool1',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		x = slim.conv2d(scope='decoder/conv2',inputs=x,num_outputs=64,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)		

		x = slim.max_pool2d(scope='decoder/pool2',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		x = slim.conv2d(scope='decoder/conv3',inputs=x,num_outputs=128,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)		

		x = slim.max_pool2d(scope='decoder/pool3',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		x = tf.reshape(x,[-1.8*8*128])

		x = tf.concat([self.input_nlcd,self.image_feature_encoder,self.input_label],1)

		x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer,scope='encoder/fc_1')
		x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer, scope='encoder/fc_2')
		#x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='encoder/fc_3')

		x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)

		z_miu = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='encoder/z_miu')
		z_logvar = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='encoder/z_logvar')

		############## Sample_z ###############

		eps = tf.random_normal(shape=tf.shape(z_miu))
		self.sample_z = z_miu + tf.exp(z_logvar / 2) * eps

		############## P(X|z) ###############

		# x = slim.conv2d(scope='decoder/conv1',inputs=self.input_image,num_outputs=16,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# x = slim.conv2d(scope='decoder/conv2',inputs=x,num_outputs=32,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# x = slim.max_pool2d(scope='decoder/pool1',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		# x = slim.conv2d(scope='decoder/conv3',inputs=x,num_outputs=64,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# x = slim.conv2d(scope='decoder/conv4',inputs=x,num_outputs=128,kernel_size=[3,3],stride=[1,1],
		# 	normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		# x = slim.max_pool2d(scope='decoder/pool2',inputs=x,kernel_size=[2,2],stride=[2,2],padding='SAME')

		# flatten_hist = tf.reshape(x,[-1,128*7*7])

		# flatten_hist = slim.fully_connected(flatten_hist, 128,weights_regularizer=weights_regularizer,scope='decoder/flatten_hist_1')

		# flatten_hist = slim.fully_connected(flatten_hist, 25,weights_regularizer=weights_regularizer,scope='decoder/flatten_hist_2')

		# self.image_feature_decoder = flatten_hist

		conv1 = slim.conv2d(scope='decoder/conv1',inputs=self.input_image,num_outputs=16,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		conv2 = slim.conv2d(scope='decoder/conv2',inputs=conv1,num_outputs=16,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		pool1 = slim.max_pool2d(scope='decoder/pool1',inputs=tf.concat([self.input_image,conv2],3),kernel_size=[2,2],stride=[2,2],padding='SAME')

		conv3 = slim.conv2d(scope='decoder/conv3',inputs=pool1,num_outputs=32,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		conv4 = slim.conv2d(scope='decoder/conv4',inputs=conv3,num_outputs=32,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		pool2 = slim.max_pool2d(scope='decoder/pool2',inputs=tf.concat([pool1,conv4],3),kernel_size=[2,2],stride=[2,2],padding='SAME')

		conv5 = slim.conv2d(scope='decoder/conv5',inputs=pool2,num_outputs=64,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		conv6 = slim.conv2d(scope='decoder/conv6',inputs=conv5,num_outputs=64,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		pool3 = slim.max_pool2d(scope='decoder/pool3',inputs=tf.concat([pool2,conv6],3),kernel_size=[2,2],stride=[2,2],padding='SAME')

		conv7 = slim.conv2d(scope='decoder/conv7',inputs=pool3,num_outputs=128,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		conv8 = slim.conv2d(scope='decoder/conv8',inputs=conv7,num_outputs=128,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		pool4 = slim.max_pool2d(scope='decoder/pool4',inputs=tf.concat([pool3,conv8],3),kernel_size=[2,2],stride=[2,2],padding='SAME')

		conv9 = slim.conv2d(scope='decoder/conv9',inputs=pool4,num_outputs=256,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		conv10 = slim.conv2d(scope='decoder/conv10',inputs=conv9,num_outputs=256,kernel_size=[3,3],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		pool5 = slim.max_pool2d(scope='decoder/pool5',inputs=tf.concat([pool4,conv10],3),kernel_size=[4,4],stride=[4,4],padding='SAME')

		conv11 = slim.conv2d(scope='decoder/conv11',inputs=pool5,num_outputs=25,kernel_size=[1,1],stride=[1,1],
			normalizer_fn=slim.batch_norm,normalizer_params=batch_norm_params,weights_regularizer = weights_regularizer)

		flatten_hist = tf.reshape(conv11,[-1,25])

		self.image_feature_decoder = flatten_hist

		x = tf.concat([self.input_nlcd,self.image_feature_decoder,self.sample_z],1)

		x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer,scope='decoder/fc_1')
		x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer, scope='decoder/fc_2')
		#x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='decoder/fc_3')

		x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)
		
		self.logits = self.sample_z+slim.fully_connected(x, 100, activation_fn=None, weights_regularizer=weights_regularizer,scope='decoder/logits')

		self.output = tf.sigmoid(self.logits,name='decoder/output')

		# E[log P(X|z)]
		self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.input_label), 1))
		tf.summary.scalar('recon_loss',self.recon_loss)
		
		# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
		self.kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_miu**2 - 1. - z_logvar, 1))
		tf.summary.scalar('kl_loss',self.kl_loss)

		# VAE loss
		self.vae_loss = self.recon_loss + self.kl_loss
		slim.losses.add_loss(self.vae_loss)
		tf.summary.scalar('vae_loss',self.vae_loss)
		
		# l2 loss
		self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())
		tf.summary.scalar('l2_loss',self.l2_loss)

		#total loss
		self.total_loss = slim.losses.get_total_loss()
		tf.summary.scalar('total_loss',self.total_loss)