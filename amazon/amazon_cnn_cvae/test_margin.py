import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import vae_margin
import get_data
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score



FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir','./summary_test','path to store summary_dir')
tf.app.flags.DEFINE_integer('batch_size',4048,'number of pictures in one batch')
tf.app.flags.DEFINE_integer('z_dim',17,'z dimention')
tf.app.flags.DEFINE_string('checkpoint_path', './model/fc_model-2950','The path to a checkpoint from which to fine-tune.')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')



def log_likelihood(all_label,all_output):
	loss = all_label*np.log(all_output)+(1-all_label)*np.log(1-all_output)
	loss = np.sum(loss,axis=1)
	loss = np.average(loss)
	return loss

def tang_reduce(output,label):
	tang=output**label*(1.0-output)**(1.0-label)
	tang=np.multiply.reduce(tang,1)
	return tang

def main(_):


	print 'reading npy...'

	data = np.load('../1st.npy')

	# test_order = np.load('../test.npy')

	# jpg_list = np.load('../nlcd+vae+image64/input_images_64.npy')

	test_jpg = np.load('../amazon_data/test_image64.npy')
	test_order = range(len(test_jpg))
	test_label = np.load('../amazon_data/test_label.npy')

	print 'reading finished'

	sess = tf.Session()

	print 'building network...'

	hg = vae_margin.vae(is_training=True)
	global_step = tf.Variable(0,name='global_step',trainable=False)

	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph)

	saver = tf.train.Saver(max_to_keep=None)
	saver.restore(sess,FLAGS.checkpoint_path)
	print 'restoring from '+FLAGS.checkpoint_path


	def test_step():

		print 'Testing...'

		all_recon_loss = 0

		all_output = []
		all_label = []
		
		batch_size = FLAGS.batch_size
		for i in range(int(len(test_order)/batch_size)):

			input_image = get_data.get_jpg_test(test_jpg,test_order[batch_size*i:batch_size*(i+1)])/128.0
			input_label = get_data.get_label(test_label,test_order[batch_size*i:batch_size*(i+1)])

			feed_dict={}
			feed_dict[hg.input_image]=input_image
			feed_dict[hg.input_label]=input_label
			feed_dict[hg.keep_prob]=1.0


			output= sess.run(hg.output,feed_dict)

			for i in output:
				all_output.append(i)
			for i in input_label:
				all_label.append(i)

		all_output = np.array(all_output)
		all_label = np.array(all_label)
		

		all_output=np.reshape(all_output,(-1,17))
		all_label=np.reshape(all_label,(-1,17))

		return all_output,all_label
	
	all_output = 0.0
	all_label = 0.0
	all_fuck = 0.0

	num = 100.0

	for i in range(int(num)):
		print i
		output,label= test_step()
		#all_output.append(output)
		all_output+=output
		all_label=label
		#fuck = np.multiply.reduce(label,1)
		fuck = tang_reduce(output,label)
		all_fuck+=fuck

	# tang = 0.0
	# for i in all_output:
	# 	tang+=i
	# tang = tang/num
	tang = all_output/num
	fuck = all_fuck/num
	epsilon = 1e-6
	#float_labels = tf.cast(labels, tf.float32)
	loss = all_label * np.log(tang + epsilon) + (1 - all_label) * np.log(1 - tang + epsilon)
	loss = np.sum(loss,axis=1)
	loss = np.mean(loss)
	fuck = np.mean(np.log(fuck))
	#print 'another final:'+str(loss)
	auc = roc_auc_score(all_label,all_output)		
	tang=np.reshape(tang,(-1))
	all_label=np.reshape(all_label,(-1))

	np.save('all_output_1wavg.npy',tang)
	np.save('all_label.npy',all_label)

	ap = average_precision_score(all_label,tang)

	time_str = datetime.datetime.now().isoformat()
	new_auc = roc_auc_score(all_label,tang)

	print 'margin results'
	tempstr = "{}: auc {:g}, ap {:g}, recon_loss {:g}, new_auc {:g}".format(time_str, auc, ap, loss, new_auc)
	print(tempstr)

	print 'average log sigma:'+str(fuck)


				

if __name__=='__main__':
	tf.app.run()