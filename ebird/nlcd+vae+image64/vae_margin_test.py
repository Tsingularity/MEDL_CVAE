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
		batch_size = FLAGS.batch_size

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


		############## Sample_z ###############

		eps = tf.random_normal(shape=[batch_size,z_dim])
		# self.sample_z = z_miu + tf.exp(z_logvar / 2) * eps
		self.sample_z = eps

		############## P(X|z) ###############

		x = tf.concat([self.input_nlcd,self.image_feature_decoder,self.sample_z],1)

		x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer,scope='decoder/fc_1')
		x = slim.fully_connected(x, 1024,weights_regularizer=weights_regularizer, scope='decoder/fc_2')
		x = slim.fully_connected(x, 512,weights_regularizer=weights_regularizer, scope='decoder/fc_3')

		x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)
		
		self.logits = slim.fully_connected(x, 100, activation_fn=None, weights_regularizer=weights_regularizer,scope='decoder/logits')

		self.output = tf.sigmoid(self.logits,name='decoder/output')