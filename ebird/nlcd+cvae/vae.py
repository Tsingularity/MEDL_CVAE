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
		
		self.input_nlcd = tf.placeholder(dtype=tf.float32,shape=[None,15],name='input_nlcd')
		
		self.input_label = tf.placeholder(dtype=tf.float32,shape=[None,100],name='input_label')

		self.keep_prob = tf.placeholder(tf.float32)

		weights_regularizer=slim.l2_regularizer(FLAGS.weight_decay)

		############## Q(z|X) ###############

		input_x = tf.concat([self.input_nlcd,self.input_label],1)

		x = slim.fully_connected(input_x, 115,weights_regularizer=weights_regularizer,scope='encoder/fc_1')
		x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer, scope='encoder/fc_2')
		x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer, scope='encoder/fc_3')

		#dropout
		#x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)

		z_miu = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='encoder/z_miu')
		z_logvar = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='encoder/z_logvar')

		condition = tf.concat([self.input_nlcd],1)

		x = slim.fully_connected(condition, 115,weights_regularizer=weights_regularizer,scope='condition/fc_1')
		x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer, scope='condition/fc_2')
		x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer, scope='condition/fc_3')
		
		condition_miu = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='condition/z_miu')
		condition_logvar = slim.fully_connected(x, z_dim, activation_fn=None, weights_regularizer=weights_regularizer,scope='condition/z_logvar')		

		############## Sample_z ###############

		eps = tf.random_normal(shape=tf.shape(z_miu))
		sample_z = z_miu + tf.exp(z_logvar / 2) * eps

		############## P(X|z) ###############

		x = tf.concat([self.input_nlcd,sample_z],1)
		x = slim.fully_connected(x, 115,weights_regularizer=weights_regularizer,scope='decoder/fc_1')
		x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer, scope='decoder/fc_2')
		x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer, scope='decoder/fc_3')
		
		#dropout
		x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)
		
		self.logits = slim.fully_connected(x, 100, activation_fn=None, weights_regularizer=weights_regularizer,scope='decoder/logits')

		self.output = tf.sigmoid(self.logits,name='decoder/output')

		# E[log P(X|z)]
		self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.input_label), 1))
		tf.summary.scalar('recon_loss',self.recon_loss)
		
		# D_KL(Q(z|X) || P(z|X)); calculate in closed form as both dist. are Gaussian
		#self.kl_loss = tf.reduce_mean(0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_miu**2 - 1. - z_logvar, 1))
		self.kl_loss = tf.reduce_mean(gaussian_kld(z_miu,z_logvar,condition_miu,condition_logvar))
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





		# self.g_output = tf.sigmoid(x)

		# self.ce_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label,logits=x),1))

		# tf.summary.scalar('ce_loss',self.ce_loss)

		# slim.losses.add_loss(self.ce_loss)		

		# self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())

		# tf.summary.scalar('l2_loss',self.l2_loss)

		# self.total_loss = slim.losses.get_total_loss()

		# tf.summary.scalar('total_loss',self.total_loss)

		# self.output = tf.sigmoid(x)

		