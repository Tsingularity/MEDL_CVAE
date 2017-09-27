import numpy as np
from sklearn.multioutput import ClassifierChain
from sklearn.linear_model import LogisticRegression
import get_data

def loglikehood(pre,label):

	# tang = -label*np.log(pre+1e-8)-(1.0-label)*np.log(1.0-pre+1e-8)
	# tang = np.mean(np.sum(tang,axis=1))
	
	tang = []
	for i in range(len(label)):
		temp = []
		for j in range(len(label[i])):
			if label[i][j] == 1:
				temp.append(np.log(pre[i][j]))
			if label[i][j] == 0:
				temp.append(np.log(1-pre[i][j]))
		tang.append(temp)
	tang = np.array(tang)
	tang = np.mean(np.sum(tang,axis=1))
	return tang

def log_likelihood(all_label,all_output):
	loss = all_label*np.log(all_output)+(1-all_label)*np.log(1-all_output)
	loss = np.sum(loss,axis=1)
	loss = np.average(loss)
	return loss

def cc():


	print 'reading npy...'

	data = np.load('../data/1st.npy')
	feature_data = np.load('dnn_feature.npy')

	train_order = np.load('../data/train.npy')
	validation_order = np.load('../data/validation.npy')
	test_order = np.load('../data/test.npy')

	train_nlcd = get_data.get_feature(feature_data,train_order)
	train_label = get_data.get_label(data,train_order)

	test_nlcd = get_data.get_feature(feature_data,test_order)
	test_label = get_data.get_label(data,test_order)

	print 'chaining'
	# Fit an ensemble of logistic regression classifier chains and take the
	# take the average prediction of all the chains.
	
	chains = []
	for i in range(10):
		chains.append(ClassifierChain(LogisticRegression(), order='random', random_state=i))
		#chains.append(ClassifierChain(LogisticRegression()))
		#chains.append(ClassifierChain(LogisticRegression(), order=range(100), random_state=i))

	
	fuck = 0
	for chain in chains:
		print fuck+1
		chain.fit(train_nlcd, train_label)
		fuck+=1

	print 'testing'
	# Y_pred_chains = np.array([chain.predict(X_test) for chain in
	#                           chains])
	# # chain_jaccard_scores = [jaccard_similarity_score(Y_test, Y_pred_chain >= .5)
	#                         for Y_pred_chain in Y_pred_chains]

	# Y_pred_ensemble = Y_pred_chains.mean(axis=0)
	# ensemble_jaccard_score = jaccard_similarity_score(Y_test,
	#                                                   Y_pred_ensemble >= .5)

	# model_scores = [ovr_jaccard_score] + chain_jaccard_scores
	# model_scores.append(ensemble_jaccard_score)
	scores = []
	for chain in chains:
		pre = chain.predict_proba(test_nlcd)
		#np.save('pre.npy',pre)
		chain_score = log_likelihood(test_label,pre)
		print chain_score
		scores.append(chain_score)
	scores = np.array(scores)
	print 'mean:'
	print np.mean(scores)


if __name__=='__main__':
	cc()