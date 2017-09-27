import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import datetime
import resnet
import get_data
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir','./model/','path to store model')
tf.app.flags.DEFINE_string('summary_dir','./summary','path to store summary_dir')
tf.app.flags.DEFINE_integer('batch_size',256,'number of pictures in one batch')
tf.app.flags.DEFINE_float('learning_rate',0.001,'initial learning rate')

tf.app.flags.DEFINE_integer('max_epoch',200,'max epoch to train')
tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('save_epoch',1.0,'epochs to save model')

def MakeSummary(name, value):
  """Creates a tf.Summary proto with the given name and value."""
  summary = tf.Summary()
  val = summary.value.add()
  val.tag = str(name)
  val.simple_value = float(value)
  return summary


def main(_):


	print 'reading npy...'

	data = np.load('../data/1st.npy')
	# jpg_list=[]
	# for i in range(len(data)):
	# 	jpg_list.append(str(i)+'.jpg')

	#jpg_list = np.load('64bin.npy')
	jpg_list = np.load('../data/input_images_64.npy')
	train_order = np.load('../data/train.npy')
	validation_order = np.load('../data/validation.npy')

	one_epoch_iter = len(train_order)/FLAGS.batch_size
	print 'reading finished'

	sess = tf.Session()

	print 'building network...'
	hg = resnet.resnet(is_training=True)
	global_step = tf.Variable(0,name='global_step',trainable=False)

	learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step,0.5*FLAGS.max_epoch*len(train_order)/FLAGS.batch_size,0.1,staircase=True)
	tf.summary.scalar('learning_rate', learning_rate)

	optimizer = tf.train.AdamOptimizer(learning_rate)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(hg.total_loss,global_step=global_step)
	merged_summary = tf.summary.merge_all()
	
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph)

	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver(max_to_keep=2)
	print 'building finished'

	

	def train_step(input_image,input_nlcd,input_label):
		feed_dict={}
		feed_dict[hg.input_image]=input_image
		feed_dict[hg.input_nlcd]=input_nlcd
		feed_dict[hg.input_label]=input_label
		feed_dict[hg.keep_prob]=0.5



		temp,step,ce_loss,l2_loss,total_loss,summary,output= sess.run([train_op,global_step,hg.ce_loss,hg.l2_loss,hg.total_loss,merged_summary,hg.output],feed_dict)
		
		summary_writer.add_summary(summary,step)

		return output,ce_loss,l2_loss,total_loss

	def validation_step(current_step):

		print 'Validating...'

		all_ce_loss = 0
		all_l2_loss = 0
		all_total_loss = 0
		all_output = []
		all_label = []
		
		for i in range(int(len(validation_order)/18)):

			input_image = get_data.get_jpg_test(jpg_list,validation_order[18*i:18*(i+1)])/128.0
			input_label = get_data.get_label(data,validation_order[18*i:18*(i+1)])
			input_nlcd = get_data.get_nlcd(data,validation_order[18*i:18*(i+1)])


			feed_dict={}
			feed_dict[hg.input_image]=input_image
			feed_dict[hg.input_nlcd]=input_nlcd
			feed_dict[hg.input_label]=input_label
			feed_dict[hg.keep_prob]=1.0

			ce_loss,l2_loss,total_loss,output= sess.run([hg.ce_loss,hg.l2_loss,hg.total_loss,hg.output],feed_dict)
			all_ce_loss += ce_loss
			all_l2_loss += l2_loss
			all_total_loss += total_loss
			for i in output:
				all_output.append(i)
			for i in input_label:
				all_label.append(i)

		all_output = np.array(all_output)
		all_label = np.array(all_label)
		auc = roc_auc_score(all_label,all_output)
		ce_loss = all_ce_loss/283.0
		l2_loss = all_l2_loss/283.0
		total_loss = all_total_loss/283.0

		all_output=np.reshape(all_output,(-1))
		all_label=np.reshape(all_label,(-1))
		ap = average_precision_score(all_label,all_output)

		time_str = datetime.datetime.now().isoformat()

		tempstr = "{}: auc {:g}, ap {:g}, ce_loss {:g}, l2_loss {:g}, total_loss {:g}".format(time_str, auc, ap, ce_loss, l2_loss, total_loss)
		print(tempstr)

		summary_writer.add_summary(MakeSummary('validation/auc',auc),current_step)
		summary_writer.add_summary(MakeSummary('validation/ap',ap),current_step)
		summary_writer.add_summary(MakeSummary('validation/ce_loss',ce_loss),current_step)

		return ce_loss

	best_ce_loss = 10000
	best_iter = 0
	smooth_ce_loss = 0
	smooth_l2_loss = 0
	smooth_total_loss = 0
	temp_label=[]	
	temp_output=[]
	for one_epoch in range(FLAGS.max_epoch):
		
		print('epoch '+str(one_epoch+1)+' starts!')
		
		np.random.shuffle(train_order)
		
		for i in range(int(len(train_order)/float(FLAGS.batch_size))):
			
			start = i*FLAGS.batch_size
			end = (i+1)*FLAGS.batch_size

			#print 'hehehehe'
			input_image = get_data.get_jpg_train(jpg_list,train_order[start:end])/128.0
			input_label = get_data.get_label(data,train_order[start:end])
			input_nlcd = get_data.get_nlcd(data,train_order[start:end])

			#print 'hahaha'

			# np.save('debug_image.npy',input_image)
			# np.save('debug_label.npy',input_label)
			# return


			output,ce_loss,l2_loss,total_loss = train_step(input_image,input_nlcd,input_label)
			
			smooth_ce_loss+=ce_loss
			smooth_l2_loss+=l2_loss
			smooth_total_loss+=total_loss
			
			temp_label.append(input_label)
			temp_output.append(output)

			current_step = tf.train.global_step(sess,global_step)

			#print current_step

			if current_step%10==0:

				ce_loss=smooth_ce_loss/10.0
				l2_loss=smooth_l2_loss/10.0
				total_loss=smooth_total_loss/10.0
				
				temp_output = np.reshape(np.array(temp_output),(-1))
				temp_label = np.reshape(np.array(temp_label),(-1))
				ap = average_precision_score(temp_label,temp_output)

				temp_output = np.reshape(temp_output,(-1,100))
				temp_label = np.reshape(temp_label,(-1,100))
				
				try:
					auc = roc_auc_score(temp_label,temp_output)

				except ValueError:
					print 'ytrue error for auc'
				
				else:
					time_str = datetime.datetime.now().isoformat()
					tempstr = "{}: step {}, auc {:g}, ap {:g}, ce_loss {:g}, l2_loss {:g}, total_loss {:g}".format(time_str, current_step, auc, ap,ce_loss, l2_loss, total_loss)
					print(tempstr)

					summary_writer.add_summary(MakeSummary('train/auc',auc),current_step)
					summary_writer.add_summary(MakeSummary('train/ap',ap),current_step)

				temp_output=[]
				temp_label=[]
				smooth_ce_loss = 0
				smooth_l2_loss = 0
				smooth_total_loss = 0
			
			if current_step%int(one_epoch_iter*FLAGS.save_epoch)==0:
				ce_loss = validation_step(current_step)
				if ce_loss<best_ce_loss:
					print 'currently the validation ce_loss is less the previous best one!!!'
					best_ce_loss=ce_loss
					best_iter=current_step
					print 'saving model'
					path = saver.save(sess,FLAGS.model_dir+'resnet_model',global_step=current_step)
					print 'have saved model to '+path

	print 'warmup training has been finished !'
	print 'the best model iter is '+str(best_iter)
	print 'the best ce_loss on validation is '+str(best_ce_loss)


if __name__=='__main__':
	tf.app.run()