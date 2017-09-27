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
tf.app.flags.DEFINE_string('summary_dir','./summary','path to store summary_dir')
tf.app.flags.DEFINE_integer('batch_size',512,'number of pictures in one batch')
tf.app.flags.DEFINE_float('learning_rate',0.0001,'initial learning rate')

tf.app.flags.DEFINE_integer('max_epoch',600,'max epoch to train')
tf.app.flags.DEFINE_integer('z_dim',100,'z dimention')

tf.app.flags.DEFINE_float('weight_decay',0.00001,'weight_decay')
tf.app.flags.DEFINE_float('save_epoch',2.0,'epochs to save model')

tf.app.flags.DEFINE_string('checkpoint_path', './model/fc_model-477','The path to a checkpoint from which to fine-tune.')


def MakeSummary(name, value):
	"""Creates a tf.Summary proto with the given name and value."""
	summary = tf.Summary()
	val = summary.value.add()
	val.tag = str(name)
	val.simple_value = float(value)
	return summary

def main(_):


	print 'reading npy...'

	data = np.load('../1st.npy')
	jpg_list = np.load('64bin.npy')
	train_order = np.load('../train.npy')
	validation_order = np.load('../validation.npy')

	one_epoch_iter = len(train_order)/FLAGS.batch_size
	print 'reading finished'

	sess = tf.Session()

	print 'building network...'

	hg = vae.vae(is_training=True)
	global_step = tf.Variable(0,name='global_step',trainable=False)

	learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step,0.5*FLAGS.max_epoch*len(train_order)/FLAGS.batch_size,1.0,staircase=True)
	tf.summary.scalar('learning_rate', learning_rate)

	optimizer = tf.train.AdamOptimizer(learning_rate)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		train_op = optimizer.minimize(hg.total_loss,global_step=global_step)
	merged_summary = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(FLAGS.summary_dir,sess.graph)

	sess.run(tf.initialize_all_variables())
	saver = tf.train.Saver(max_to_keep=None)
	print 'building finished'

	

	#def train_step(input_nlcd,input_label,smooth_ce_loss,smooth_l2_loss,smooth_total_loss):
	def train_step(input_label,input_nlcd,input_image):

		feed_dict={}
		feed_dict[hg.input_nlcd]=input_nlcd
		feed_dict[hg.input_label]=input_label
		feed_dict[hg.input_image]=input_image
		feed_dict[hg.keep_prob]=0.8


		temp,step,recon_loss,kl_loss,vae_loss,l2_loss,total_loss,summary,output = sess.run([train_op,global_step,hg.recon_loss,hg.kl_loss,hg.vae_loss,hg.l2_loss,hg.total_loss,merged_summary,hg.output],feed_dict)
		time_str = datetime.datetime.now().isoformat()
		summary_writer.add_summary(summary,step)

		return output, recon_loss, kl_loss, vae_loss, l2_loss, total_loss

	def validation_step():

		print 'Validating...'

		all_recon_loss = 0
		all_kl_loss = 0
		all_vae_loss = 0
		all_l2_loss = 0
		all_total_loss = 0

		all_output = []
		all_label = []
		
		for i in range(int(len(validation_order)/18)):

			input_image = get_data.get_jpg_test(jpg_list,validation_order[18*i:18*(i+1)])
			input_nlcd = get_data.get_nlcd(data,validation_order[18*i:18*(i+1)])
			input_label = get_data.get_label(data,validation_order[18*i:18*(i+1)])

			feed_dict={}
			feed_dict[hg.input_nlcd]=input_nlcd
			feed_dict[hg.input_label]=input_label
			feed_dict[hg.input_image]=input_image
			feed_dict[hg.keep_prob]=1.0

			recon_loss,kl_loss,vae_loss,l2_loss,total_loss,output= sess.run([hg.recon_loss,hg.kl_loss,hg.vae_loss,hg.l2_loss,hg.total_loss,hg.output],feed_dict)
			
			all_recon_loss += recon_loss
			all_kl_loss += kl_loss
			all_vae_loss += vae_loss
			all_l2_loss += l2_loss
			all_total_loss += total_loss
			
			for i in output:
				all_output.append(i)
			for i in input_label:
				all_label.append(i)

		all_output = np.array(all_output)
		all_label = np.array(all_label)
		#average_precision = average_precision_score(all_label,all_output)
		auc = roc_auc_score(all_label,all_output)
		
		recon_loss = all_recon_loss/283.0
		kl_loss = all_kl_loss/283.0
		vae_loss = all_vae_loss/283.0
		l2_loss = all_l2_loss/283.0
		total_loss = all_total_loss/283.0

		all_output=np.reshape(all_output,(-1))
		all_label=np.reshape(all_label,(-1))
		ap = average_precision_score(all_label,all_output)

		time_str = datetime.datetime.now().isoformat()

		tempstr = "{}: auc {:g}, ap {:g}, recon_loss {:g}, kl_loss {:g}, vae_loss {:g}, l2_loss {:g}, total_loss {:g}".format(time_str, auc, ap, recon_loss, kl_loss, vae_loss, l2_loss, total_loss)
		print(tempstr)

		summary_writer.add_summary(MakeSummary('validation/auc',auc),current_step)
		summary_writer.add_summary(MakeSummary('validation/ap',ap),current_step)
		summary_writer.add_summary(MakeSummary('validation/recon_loss',recon_loss),current_step)
		summary_writer.add_summary(MakeSummary('validation/kl_loss',kl_loss),current_step)
		summary_writer.add_summary(MakeSummary('validation/vae_loss',vae_loss),current_step)

		return vae_loss


	best_vae_loss = 10000
	best_iter = 0
	smooth_recon_loss=0.0
	smooth_kl_loss=0.0
	smooth_vae_loss=0.0
	smooth_l2_loss=0.0
	smooth_total_loss=0.0
	temp_label=[]	
	temp_output=[]

	for one_epoch in range(FLAGS.max_epoch):
		
		print('epoch '+str(one_epoch+1)+' starts!')
		np.random.shuffle(train_order)
		
		for i in range(int(len(train_order)/float(FLAGS.batch_size))):
			
			start = i*FLAGS.batch_size
			end = (i+1)*FLAGS.batch_size

			input_image = get_data.get_jpg_test(jpg_list,train_order[start:end])
			input_nlcd = get_data.get_nlcd(data,train_order[start:end])
			input_label = get_data.get_label(data,train_order[start:end])

			output, recon_loss, kl_loss, vae_loss, l2_loss, total_loss = train_step(input_label,input_nlcd,input_image)
			
			smooth_recon_loss+=recon_loss
			smooth_kl_loss+=kl_loss
			smooth_vae_loss+=vae_loss
			smooth_l2_loss+=l2_loss
			smooth_total_loss+=total_loss
			
			temp_label.append(input_label)
			temp_output.append(output)

			current_step = tf.train.global_step(sess,global_step)

			if current_step%10==0:

				recon_loss=smooth_recon_loss/10.0
				kl_loss=smooth_kl_loss/10.0
				vae_loss=smooth_vae_loss/10.0
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
					tempstr = "{}: step {}, auc {:g}, ap {:g}, recon_loss {:g}, kl_loss {:g}, vae_loss {:g}, l2_loss {:g}, total_loss {:g}".format(time_str, current_step, auc, ap, recon_loss, kl_loss, vae_loss, l2_loss, total_loss)
					print(tempstr)
					summary_writer.add_summary(MakeSummary('train/auc',auc),current_step)
					summary_writer.add_summary(MakeSummary('train/ap',ap),current_step)

				temp_output=[]
				temp_label=[]
				smooth_recon_loss = 0
				smooth_kl_loss = 0
				smooth_vae_loss = 0
				smooth_l2_loss = 0
				smooth_total_loss = 0

			if current_step%int(one_epoch_iter*FLAGS.save_epoch)==0:
				vae_loss = validation_step()
				if vae_loss<best_vae_loss:
					print 'currently the vae_loss is over the previous best one!!!'
					best_vae_loss=vae_loss
					best_iter = current_step
				print 'saving model'
				path = saver.save(sess,FLAGS.model_dir+'fc_model',global_step=current_step)
				print 'have saved model to '+path

	print 'training has been finished !'
	print 'the best vae_loss on validation is '+str(best_vae_loss)
	print 'the best checkpoint is '+str(best_iter)


if __name__=='__main__':
	tf.app.run()