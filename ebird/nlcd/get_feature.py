import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import fc_test
import get_data
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir','./model/','path to store model')
tf.app.flags.DEFINE_string('summary_dir','./summary_test','path to store summary_dir')
tf.app.flags.DEFINE_integer('batch_size',512,'number of pictures in one batch')
tf.app.flags.DEFINE_float('learning_rate',0.01,'initial learning rate')

tf.app.flags.DEFINE_integer('max_epoch',100,'max epoch to train')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('save_epoch',2.0,'epochs to save model')

tf.app.flags.DEFINE_string('checkpoint_path', './model/fc_model-7742','The path to a checkpoint from which to fine-tune.')


def log_likelihood(all_label,all_output):
	loss = all_label*np.log(all_output)+(1-all_label)*np.log(1-all_output)
	loss = np.sum(loss,axis=1)
	loss = np.average(loss)
	return loss

def main(_):


	print 'reading npy...'

	data = np.load('../1st.npy')

	#test_order = np.load('../test.npy')
	test_order = []
	for i in range(len(data)):
		test_order.append(i)

	print 'reading finished'

	sess = tf.Session()

	print 'building network...'

	hg = fc_test.fc(is_training=False)
	global_step = tf.Variable(0,name='global_step_new',trainable=False)

	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph)

	saver = tf.train.Saver(max_to_keep=None)
	saver.restore(sess,FLAGS.checkpoint_path)
	print 'restoring from '+FLAGS.checkpoint_path


	def test_step():

		print 'testing...'

		all_output = []
		
		
		for i in range(int(len(test_order)/17)):

			input_nlcd = get_data.get_nlcd(data,test_order[17*i:17*(i+1)])
			input_label = get_data.get_label(data,test_order[17*i:17*(i+1)])

			feed_dict={}
			feed_dict[hg.input_nlcd]=input_nlcd
			feed_dict[hg.input_label]=input_label
			feed_dict[hg.keep_prob]=1.0


			output= sess.run(hg.feature,feed_dict)
			for i in output:
				all_output.append(i)

		all_output = np.array(all_output)
		np.save('dnn_feature.npy',all_output)

	
	test_step()
				

if __name__=='__main__':
	tf.app.run()