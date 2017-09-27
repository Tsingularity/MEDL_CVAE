import numpy as np
from numpy import linalg as LA

a = np.load('embed_17.npy')

matrix = []

for i in range(17):
	temp = []
	for j in range(17):
		cos = np.dot(a[i],a[j])/(LA.norm(a[i])*LA.norm(a[j]))
		#cos = (LA.norm(a[i]))**2+(LA.norm(a[j]))**2-2*LA.norm(a[j])*LA.norm(a[i])*np.dot(a[i],a[j])/(LA.norm(a[i])*LA.norm(a[j]))
		temp.append(cos)
	matrix.append(temp)

np.save('matrix.npy',matrix)


labels = [
	'agriculture',
	'artisinal_mine',
	'bare_ground',
	'blooming',
	'blow_down',
	'clear',
	'cloudy',
	'conventional_mine',
	'cultivation',
	'habitation',
	'haze',
	'partly_cloudy',
	'primary',
	'road',
	'selective_logging',
	'slash_burn',
	'water',
]

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

K_labels = []

# for i in top_50_labels:
#     row = []
#     for j in top_50_labels:
#         # find all records that have label `i` in them
#         i_occurs = [x for x in textual_labels_nested if i in x]
#         # how often does j occur in total in them?
#         j_and_i_occurs = [x for x in i_occurs if j in x]
#         k = 1.0*len(j_and_i_occurs)/len(i_occurs)
#         row.append(k)
#     K_labels.append(row)

K_labels = np.array(matrix)
K_labels = pd.DataFrame(K_labels)
K_labels.columns = labels
K_labels.index = labels

plt.figure(figsize=(12,8))
sns.heatmap(K_labels,cmap="gray_r",annot=True)
# probability of observing column label given row label
plt.title('P(column|row)')
plt.show()