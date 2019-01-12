import sys
sys.path.insert(0, 'caffe_rnet/python')
import caffe

import numpy as np

import random
from skimage.io import imread

#_weights = 'icnet_cityscapes_train_30k.caffemodel'

caffe.set_mode_gpu();
caffe.set_device(1)
solver_path = 'solver.prototxt'

mean = [43, 46, 5]
solver = caffe.get_solver(solver_path);
#solver.net.copy_from(_weights);

file_img = open('/home/ubuntu/Datasets/TUMOR_SUR/train_ts.txt', 'r')
train_img ={};

idx = 0;
for line in file_img:
    train_img[idx] = line[0:-1];
    idx = idx + 1;

batch_size = 1;
#print solver.net.blobs['data'].shape
#solver.net.blobs['data'].reshape(batch_size,3,270,270);
#solver.net.blobs['label'].reshape(batch_size,1,270,270);
sample = len(train_img);
N = range(sample);
ite_total = sample/batch_size;
for epochs in range(0,4000):
    random.shuffle(N);
    ite_idx = 0;
    for ite in range(0,ite_total):
        for batch_idx in xrange(batch_size):
            img = imread(train_img[N[ite_idx]]+ '.png');
            img = img - mean;
            img = img.transpose()
            lab = imread(train_img[N[ite_idx]]+ '-seg.png').transpose();
	    #IMG = np.zeros(shape=(3, 480, 480))
	    #LAB = np.zeros(shape=(1, 480, 480))
	    #IMG[:, :, 105:375] = img
	    #LAB[:, :, 105:375] = lab
            img = img[:, 105:375, :]
            lab = lab[105:375, :]
            #print "IMAGE: ", img.shape
            #print "GT: ", lab.shape
            #print "Blob: ", solver.net.blobs['data'].data.shape
            solver.net.blobs['data'].data[batch_idx] = img;
            solver.net.blobs['label'].data[batch_idx] = lab;
            ite_idx = ite_idx + 1;
            break
	# PRINTING THE WEIGHTS
	#i = 0
	#for layer_name, param in solver.net.params.iteritems():
		#if i==0:
			#pass
	                #print layer_name + '\t' + 'weight: ' + str(param[0].data) + '  bias:' + str(param[1].data)
		#else:
   			#print layer_name + '\t' + 'weight: ' + str(param[0].data) + '  bias:' + str(param[1].data)
		#i += 1
        solver.step(1)
        break
    if epochs%10 == 0:
        solver.net.save('trained_models_mean/rnet_'+str(epochs)+'.caffemodel')

