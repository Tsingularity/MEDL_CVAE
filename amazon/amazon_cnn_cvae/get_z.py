import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import vae
import get_data
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir','./model/','path to store model')
tf.app.flags.DEFINE_string('summary_dir','./summary_test','path to store summary_dir')
tf.app.flags.DEFINE_integer('batch_size',512,'number of pictures in one batch')
tf.app.flags.DEFINE_float('learning_rate',0.01,'initial learning rate')

tf.app.flags.DEFINE_integer('max_epoch',200,'max epoch to train')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('save_epoch',1.0,'epochs to save model')
tf.app.flags.DEFINE_integer('z_dim',17,'z dimention')


tf.app.flags.DEFINE_string('checkpoint_path', './model/fc_model-2950','The path to a checkpoint from which to fine-tune.')


def main(_):


	print 'reading npy...'

	#data = np.load('../1st.npy')

	#jpg_list = np.load('64bin.npy')
	# jpg_list = np.load('../nlcd+vae+image64/input_images_64.npy')
	# test_order = np.load('../test.npy')

	test_jpg = np.load('../amazon_data/test_image64.npy')
	test_order = range(len(test_jpg))
	test_label = np.load('../amazon_data/test_label.npy')

	print 'reading finished'

	sess = tf.Session()

	print 'building network...'
	hg = vae.vae(is_training=True)
	global_step = tf.Variable(0,name='global_step',trainable=False)

	merged_summary = tf.summary.merge_all()
	
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph)

	saver = tf.train.Saver(max_to_keep=None)
	saver.restore(sess,FLAGS.checkpoint_path)
	print 'restoring from '+FLAGS.checkpoint_path


	def test_step():

		print 'Testing...'

		all_z = []
		
		batch_size = 23
		for i in range(int(len(test_order)/batch_size)):

			input_image = get_data.get_jpg_test(test_jpg,test_order[batch_size*i:batch_size*(i+1)])/128.0
			input_label = get_data.get_label(test_label,test_order[batch_size*i:batch_size*(i+1)])

			feed_dict={}
			feed_dict[hg.input_image]=input_image
			feed_dict[hg.input_label]=input_label
			feed_dict[hg.keep_prob]=1.0

			z_miu= sess.run(hg.output,feed_dict)

			all_z.append(z_miu)
			


		all_z = np.array(all_z)
		all_z = np.reshape(all_z,(-1,17))
		np.save('all_output.npy',all_z)
		
		
		# recon_loss = all_recon_loss/(4048/batch_size)
		# kl_loss = all_kl_loss/(4048/batch_size)
		# vae_loss = all_vae_loss/(4048/batch_size)
		# l2_loss = all_l2_loss/(4048/batch_size)
		# total_loss = all_total_loss/(4048/batch_size)

		# all_output=np.reshape(all_output,(-1))
		# all_label=np.reshape(all_label,(-1))
		# ap = average_precision_score(all_label,all_output)
		# auc = roc_auc_score(all_label,all_output)

		# time_str = datetime.datetime.now().isoformat()

		# tempstr = "{}: auc {:g}, ap {:g}, recon_loss {:g}, kl_loss {:g}, vae_loss {:g}, l2_loss {:g}, total_loss {:g}".format(time_str, auc, ap, recon_loss, kl_loss, vae_loss, l2_loss, total_loss)
		# print(tempstr)
		# auc = roc_auc_score(all_label,all_output)
		# print "new_auc {:g}".format(auc)

	test_step()


if __name__=='__main__':
	tf.app.run()