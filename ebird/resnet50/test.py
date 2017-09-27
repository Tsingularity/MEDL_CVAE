import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import resnet_test
import get_data
import resnet_v2
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir','./summary_test','path to store summary_dir')

#fine-tune flags
tf.app.flags.DEFINE_string('checkpoint_path', './model/resnet_model-42009','The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', None,'Comma-separated list of scopes of variables to exclude when restoring ''from a checkpoint.')
tf.app.flags.DEFINE_string('trainable_scopes', None,'Comma-separated list of scopes to filter the set of variables to train.''By default, None would train all the variables.')

def _get_init_fn():
	"""Returns a function run by the chief worker to warm-start the training.
	Note that the init_fn is only run when initializing the model during the very
	first global step.
	Returns:
	An init function run by the supervisor.
	"""
	if FLAGS.checkpoint_path is None:
		return None

	exclusions = []
	
	if FLAGS.checkpoint_exclude_scopes:
		exclusions = [scope.strip() for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

  # TODO(sguada) variables.filter_variables()
	variables_to_restore = []

	for var in slim.get_model_variables():
		
		#print var.op.name
		excluded = False
		for exclusion in exclusions:
			if var.op.name.startswith(exclusion):
				excluded = True
				break
		if not excluded:
			variables_to_restore.append(var)

	tf.logging.info('Fine-tuning from %s' % FLAGS.checkpoint_path)

	return slim.assign_from_checkpoint_fn(FLAGS.checkpoint_path,variables_to_restore,ignore_missing_vars=False)

def _get_variables_to_train():
	"""Returns a list of variables to train.
	Returns:
	A list of variables to train by the optimizer.
	"""
	if FLAGS.trainable_scopes is None:
		return tf.trainable_variables()
	else:
		scopes = [scope.strip() for scope in FLAGS.trainable_scopes.split(',')]

	variables_to_train = []
	for scope in scopes:
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
		variables_to_train.extend(variables)
	return variables_to_train

def log_likelihood(all_label,all_output):
	loss = all_label*np.log(all_output)+(1-all_label)*np.log(1-all_output)
	loss = np.sum(loss,axis=1)
	loss = np.average(loss)
	return loss



def main(_):


	print 'reading npy...'
	#trainlist, labels = read_csv.train_data()

	#jpg_list = np.load('../jpg_1st.npy')
	data = np.load('../1st.npy')
	jpg_list=[]
	for i in range(len(data)):
		jpg_list.append(str(i)+'.jpg')
	

	#train_order = np.load('../train.npy')
	test_order = np.load('../test.npy')

	sess = tf.Session()
	arg_scope = resnet_v2.resnet_arg_scope()

	print 'building network...'
	with slim.arg_scope(arg_scope):
		hg = resnet_test.resnet(is_training=False)
	init_fn = _get_init_fn()
	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph)

	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver(max_to_keep=None)
	init_fn(sess)
	print 'building finished'


	def test_step():

		print 'testing...'

		all_ce_loss = 0
		all_l2_loss = 0
		all_total_loss = 0
		all_output = []
		all_label = []
		
		for i in range(int(len(test_order)/18)):

			input_image = get_data.get_jpg_test(jpg_list,test_order[18*i:18*(i+1)])
			input_label = get_data.get_label(data,test_order[18*i:18*(i+1)])

			feed_dict={}
			feed_dict[hg.input_image]=input_image

			output= sess.run(hg.output,feed_dict)
			for i in output:
				all_output.append(i)
			for i in input_label:
				all_label.append(i)

		all_output = np.array(all_output)
		all_label = np.array(all_label)
		#average_precision = average_precision_score(all_label,all_output)
		np.save('output.npy',all_output)
		np.save('label.npy',all_label)

		auc = roc_auc_score(all_label,all_output)
		loglike = log_likelihood(all_label,all_output)

		time_str = datetime.datetime.now().isoformat()

		tempstr = "{}: auc {:g}, log_likelihood {:g}".format(time_str, auc, loglike)
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