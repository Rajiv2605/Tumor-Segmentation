# merge
from __future__ import division
import sys
sys.path.insert(0, "/home/ubuntu/rajiv/caffe_psp/python")

import caffe
from skimage.io import imread, imsave
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')

caffe.set_mode_gpu()
caffe.set_device(0)

deploy_net = '/home/ubuntu/rajiv/3D_DenseSeg/test_dense_merge.prototxt';
#mean_pixel = [43, 46, 5]
checkpoint_no = 0;
#test_limit = 3
best_dice = 0
best_model = ''
diceVals = []
epochVals = []

def dice(pred, label):
    if (np.float(np.sum(label == 1) + np.sum(pred == 1))) == 0:
        return 1
    dice_val = np.float(np.sum(pred[label==1] == 1))*2.0 / (np.float(np.sum(label == 1) + np.sum(pred == 1)));
    return dice_val

for CHPT_NO in range(108, 110, 2):
    d = 0
    weights = 'trained_models/densenet_' + str(CHPT_NO) + '.caffemodel'
    #weights = '/home/ubuntu/rajiv/ICNet_FINAL/trained_models_bs3/icnet_30_bs3.caffemodel'
    net = caffe.Net(deploy_net, weights, caffe.TEST);
    print "CHECKPOINT NUMBER: ", CHPT_NO
    idx = 0
    with open('/home/ubuntu/Datasets/ISLES2018/test.txt', 'r') as f:
        # n_img = 0
        for line in f:          
            input_image_file, ground_truth_file = line.split();
            gt = imread(ground_truth_file)
            if len(np.unique(gt))==1:
                continue
            idx += 1
            image_ori = imread(input_image_file);
            #image_ori = image_ori - mean_pixel;
            net.blobs['data'].data[0] = image_ori.transpose()
            net.forward()
            predicted = net.blobs['prob'].data[0]
            output = np.argmax(predicted, axis=0)
            #print np.unique(output)
            '''
            n_img += 1
            if n_img == test_limit:
                break
            '''
            D = dice(output.transpose(), gt)
            d += D
            #print D
            '''
            img = net.blobs['data'].data[0]
            print "Shape of image is: ", img.shape
            plt.imshow(img.transpose().astype(np.uint8))
            plt.show()
            gt = imread(ground_truth_file)
            print "Shape of gt is: ", gt.shape
            plt.imshow(gt.astype(np.uint8))
            plt.show()
            print "Shape of prediction is: ", output.shape
            plt.imshow(output.transpose().astype(np.uint8))
            plt.show()
            '''
            #break
        #break
        d = d/idx
        #print idx
        diceVals.append(d)
        epochVals.append(CHPT_NO)
        if(d>best_dice):
            best_dice = d
            best_model = weights

    print 'd: ', d
    print "Current best Model: ", best_model
print "BEST DICE ACCURACY: ", best_dice, ' and best model is: ', best_model

#plt.plot(np.array(epochVals), np.array(diceVals))
#plt.savefig('dice_results.png')

