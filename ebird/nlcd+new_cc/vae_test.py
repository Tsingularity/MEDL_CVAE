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


		all_logits = []
		all_output = []

		for i in range(100):
			if i == 0 :
				current_input_x = self.input_nlcd
			else:
				current_output = tf.concat(all_output,1)
				current_input_x = tf.concat([self.input_nlcd,current_output],1)

			x = slim.fully_connected(current_input_x, 256,weights_regularizer=weights_regularizer)
			x = slim.fully_connected(x, 256,weights_regularizer=weights_regularizer)
			x = slim.fully_connected(x, 100,weights_regularizer=weights_regularizer)

			x = slim.dropout(x,keep_prob=self.keep_prob,is_training=is_training)
			all_logits.append(slim.fully_connected(x, 1, activation_fn=None, weights_regularizer=weights_regularizer))
			all_output.append(tf.sigmoid(all_logits[i]))

		final_logits = tf.concat(all_logits,1)
		final_output = tf.sigmoid(final_logits)

		self.output = final_output
		self.ce_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label,logits=final_logits),1))


		# self.g_output = tf.sigmoid(x)

		# self.ce_loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_label,logits=x),1))

		# tf.summary.scalar('ce_loss',self.ce_loss)

		# slim.losses.add_loss(self.ce_loss)		

		# self.l2_loss = tf.add_n(slim.losses.get_regularization_losses())

		# tf.summary.scalar('l2_loss',self.l2_loss)

		# self.total_loss = slim.losses.get_total_loss()

		# tf.summary.scalar('total_loss',self.total_loss)

		# self.output = tf.sigmoid(x)

		