import numpy as np
import tensorflow as tf
from skimage.io import imread
import model
from keras import backend as K

sess = tf.Session()
K.set_session(sess)

def dice(pred, label):
    #label = label[135:270, 240:375]
    #print "Pred: ", pred.shape
    #print "Label: ", label.shape
    pred = pred[0]
    pred = np.argmax(pred)
    intersection = np.logical_and(pred, label)
    dsc = (2.0*intersection.sum())/(pred.sum()+label.sum())
    return dsc

#image = tf.placeholder("float32", [None, 270, 480, 3])
#x = model.Deeplabv3(image)

# with tf.Session() as sess:
#     # For visualising the graph
#     writer = tf.summary.FileWriter("nn_logs", sess.graph)
#     # For adding additional visualisations to Tensorboard
#     merged = tf.summary.merge_all()
# #pred = tf.cast(pred, tf.float32)

#saver = tf.train.Saver()

def test(x, epn, image):
  idx = 0
 d = 0
 #epn = 600
 with tf.Session() as sess:
  # LOADING THE SAVED MODELS
  #saver.restore(sess, "/home/ubuntu/rajiv/TENSORFLOW/tensorflow-fcn/trained_models/260/checkpoint")
  new_saver = tf.train.import_meta_graph('trained_models/' + str(epn) + '/deeplabmodel-' + str(epn) + '.meta')
  new_saver.restore(sess, tf.train.latest_checkpoint('trained_models/' + str(epn) + '/'))
  #saver.restore(sess, '/home/ubuntu/rajiv/TENSORFLOW/tensorflow-fcn/trained_models/240/checkpoint')
  #init = tf.global_variables_initializer()
  #sess.run(init)
  imgs = []
  with open('/home/ubuntu/Datasets/TUMOR_SUR/test.txt', 'r') as f:
      for line in f:
       idx += 1
          input_image_file, ground_truth_file = line.split();
          image_ori = imread(input_image_file);
          gt = imread(ground_truth_file)
          imgs.append(image_ori)
          #image_ori_tensor = tf.expand_dims(image_ori, 0)
          output = sess.run(x, feed_dict={image:imgs})
          d += dice(output, gt)
          imgs[:] = []
          if idx == 10:
           break

 print("Dice: ", d/10)
