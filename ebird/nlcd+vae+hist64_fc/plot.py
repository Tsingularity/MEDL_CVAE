import numpy as np
import matplotlib.pyplot as plt

hist = np.load('64bin.npy')

index=[0,66,256]

for i in index:
	image = hist[i]

	plt.cla()
	im=plt.imshow(image,cmap='viridis',aspect=7)
	#plt.colorbar(im,ticks=[0,1])
	plt.axis('off')
	#plt.show()
	plt.savefig(str(i)+'.png')