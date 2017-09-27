import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import resnet_test
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

tf.app.flags.DEFINE_string('checkpoint_path', './model/resnet_model-17808','The path to a checkpoint from which to fine-tune.')


def main(_):


	print 'reading npy...'

	data = np.load('../data/1st.npy')

	jpg_list = np.load('../data/64bin.npy')
	test_order = np.load('../data/test.npy')
	print 'reading finished'

	sess = tf.Session()

	print 'building network...'
	hg = resnet_test.resnet(is_training=False)
	global_step = tf.Variable(0,name='global_step',trainable=False)

	merged_summary = tf.summary.merge_all()
	
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph)

	saver = tf.train.Saver(max_to_keep=None)
	saver.restore(sess,FLAGS.checkpoint_path)
	print 'restoring from '+FLAGS.checkpoint_path


	def test_step():

		print 'Testing...'

		all_ce_loss = 0
		all_output = []
		all_label = []
		
		batch_size = 18
		for i in range(int(len(test_order)/batch_size)):

			input_image = get_data.get_jpg_test(jpg_list,test_order[batch_size*i:batch_size*(i+1)])
			input_label = get_data.get_label(data,test_order[batch_size*i:batch_size*(i+1)])
			input_nlcd = get_data.get_nlcd(data,test_order[batch_size*i:batch_size*(i+1)])


			feed_dict={}
			feed_dict[hg.input_image]=input_image
			feed_dict[hg.input_label]=input_label
			feed_dict[hg.input_nlcd]=input_nlcd
			feed_dict[hg.keep_prob]=1.0

			ce_loss,output= sess.run([hg.ce_loss,hg.output],feed_dict)
			all_ce_loss += ce_loss
			for i in output:
				all_output.append(i)
			for i in input_label:
				all_label.append(i)

		all_output = np.array(all_output)
		all_label = np.array(all_label)
		#average_precision = average_precision_score(all_label,all_output)

		loglike = all_ce_loss/(int(len(test_order)/batch_size))

		np.save('output.npy',all_output)
		np.save('label.npy',all_label)
		
		auc = roc_auc_score(all_label,all_output)
		#loglike = log_likelihood(all_label,all_output)

		time_str = datetime.datetime.now().isoformat()

		tempstr = "{}: auc {:g}, log_likelihood {:g}".format(time_str, auc,loglike)
		print(tempstr)

		all_output=np.reshape(all_output,(-1))
		all_label=np.reshape(all_label,(-1))
		ap = average_precision_score(all_label,all_output)
		auc_2 = roc_auc_score(all_label,all_output)
		print 'ap:'+str(ap)
		print 'auc_2:'+str(auc_2)

	test_step()


if __name__=='__main__':
	tf.app.run()