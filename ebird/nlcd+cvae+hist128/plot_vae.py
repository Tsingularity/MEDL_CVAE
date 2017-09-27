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
tf.app.flags.DEFINE_integer('z_dim',100,'z dimention')


tf.app.flags.DEFINE_integer('max_epoch',100,'max epoch to train')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('save_epoch',2.0,'epochs to save model')

tf.app.flags.DEFINE_string('checkpoint_path', './model/fc_model-35234','The path to a checkpoint from which to fine-tune.')


def log_likelihood(all_label,all_output):
	loss = all_label*np.log(all_output)+(1-all_label)*np.log(1-all_output)
	loss = np.sum(loss,axis=1)
	loss = np.average(loss)
	return loss

def main(_):


	print 'reading npy...'

	data = np.load('../data/1st.npy')

	#test_order = np.load('../test.npy')
	jpg_list = np.load('../data/128bin.npy')
	#jpg_list = np.load('../nlcd+vae+image64/input_images_64.npy')

	# forest_order = [0,1,2,3,12,19,38,75,79,147]

	# human_order = [238,257,726,886,888,1397,1730,1910,26834,27110]

	# ocean_order = [45,178,266,876,1516,8112,8201,8365,8495,9318,26471]

	# orders = [forest_order,human_order,ocean_order]

	orders = []
	for i in range(len(data)):
		orders.append(i)

	print 'reading finished'

	sess = tf.Session()

	print 'building network...'

	hg = vae.vae(is_training=False)
	global_step = tf.Variable(0,name='global_step',trainable=False)

	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph)

	saver = tf.train.Saver(max_to_keep=None)
	saver.restore(sess,FLAGS.checkpoint_path)
	print 'restoring from '+FLAGS.checkpoint_path


	def test_step():

		print 'Testing...'

		# names = ['forest','human','ocean']

		all_sample_z=[]
		
		batch_size = 17

		# for i in range(len(orders)):

		# 	input_image = get_data.get_jpg_test(jpg_list,orders[i])/128.0
		# 	input_nlcd = get_data.get_nlcd(data,orders[i])
		# 	input_label = get_data.get_label(data,orders[i])
		for i in range(int(len(orders)/batch_size)):

			input_image = get_data.get_jpg_test(jpg_list,orders[batch_size*i:batch_size*(i+1)])
			input_nlcd = get_data.get_nlcd(data,orders[batch_size*i:batch_size*(i+1)])
			input_label = get_data.get_label(data,orders[batch_size*i:batch_size*(i+1)])

			feed_dict={}
			feed_dict[hg.input_nlcd]=input_nlcd
			feed_dict[hg.input_image]=input_image
			feed_dict[hg.input_label]=input_label
			feed_dict[hg.keep_prob]=1.0


			#sample_z= sess.run(hg.sample_z,feed_dict)
			sample_z= sess.run(hg.condition_miu,feed_dict)

			all_sample_z.append(sample_z)

		all_sample_z = np.array(all_sample_z)

		all_sample_z = np.reshape(all_sample_z,(-1,100))

		np.save('condition_miu.npy',all_sample_z)
			
	
	test_step()
				

if __name__=='__main__':
	tf.app.run()