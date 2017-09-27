import numpy as np
from tsne import tsne
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


from sklearn.decomposition import PCA 


def gather(data,orders):
	tang = []
	for i in orders:
		tang.append(data[i])
	return np.array(tang)


pca=PCA(n_components=2)


# forest = np.load('forest.npy')
# human = np.load('human.npy')
# ocean = np.load('ocean.npy')

sample_z = np.load('sample_z.npy')
all_label = np.load("all_label.npy")
all_label = np.reshape(all_label,[-1,100])

bird_1 = all_label[:,1]


# forest_order = [0,1,2,3,12,19,38,75,79,147]

# human_order = [238,257,726,886,888,1397,1730,1910,26834,27110]

# ocean_order = [45,178,266,876,1516,8112,8201,8365,8495,9318,26471]

# forest_order = [0,1,2,3,12,19,38,75,79,147,13266,12226,12316,12341,12514,14090,14649,19237,19390,20048]

# human_forest_order = [238,257,726,886,888,1397,1730,26834,27110,13219,13147,16077,18200,18633,18849,19534,19941,19925,43478]

# human_ocean_order = [1910,16365,16366,17355,18085,19758,44284,44691,44842,45,266,876,14883,15116,50694]

# ocean_order = [178,1516,8112,8201,8365,8495,9318,26471,14503,14550,14567,15025,17391,18230,20560,20534,42954]


# forest = gather(sample_z,forest_order)
# human_forest = gather(sample_z,human_forest_order)
# human_ocean = gather(sample_z,human_ocean_order)
# ocean = gather(sample_z,ocean_order)

# embedding = np.concatenate((forest,human_forest,human_ocean,ocean),0)

size = 100

embedding = sample_z[:size]

tsne_data = tsne(embedding, 2, 100, 20,1000);

#tsne_data=pca.fit_transform(embedding)

# model = TSNE(n_components=2, random_state=10,perplexity=10,n_iter=100000)
# tsne_data = model.fit_transform(embedding) 

np.save('bird_tsne_data.npy',tsne_data)

print np.min(tsne_data[:,0])
print np.max(tsne_data[:,0])

print np.min(tsne_data[:,1])
print np.max(tsne_data[:,1])



marker = []
color = []

for i in range(size):
	if bird_1[i]==0:
		marker.append('_')
		color.append('black')
	if bird_1[i]==1:
		marker.append('+')
		color.append('red')
# for i in range(len(human_forest)):
# 	marker.append('*')
# 	color.append('green')

# for i in range(len(human_ocean)):
# 	marker.append('*')
# 	color.append('blue')

# for i in range(len(ocean)):
# 	marker.append('o')
# 	color.append('blue')

for i in range(size):
	plt.scatter(tsne_data[i, 0], tsne_data[i, 1], c=color[i], marker=marker[i],s=100)

plt.savefig('bird_tsne.png')

plt.show()

