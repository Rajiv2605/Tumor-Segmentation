import tensorflow as tf
import model
import numpy as np
from skimage.io import imread
from keras import backend as K
import os
import test

# Starting the tf session and initialising the graph built in Keras
sess = tf.Session()
K.set_session(sess)

# Common Parameters
saved_model_name = "deeplabmodel"
h = 480
w = 270
n_samples = 330
n_classes = 2
lr = 1e-3
n_epochs = 200
batch_size = 2
data_path = '/home/ubuntu/Datasets/TUMOR_SUR/train.txt'

# Returns X and y of shape [n_samples, h, w, channels] and [n_samples, n_classes] respectively
def getData():
  X = []
 y = []
 with open(data_path, 'r') as f:
  for line in f:
   img_path, gt_path = line.split()
   X.append(imread(img_path))
   GT = np.zeros([w, h, n_classes])
   gt = imread(gt_path)
   W, H, _ = GT.shape
   for i in range(W):
    for j in range(H):
     GT[i, j, gt[i, j]] = 1
   y.append(GT)

 return X, y

def train():
 #Build the network graph
 image = tf.placeholder("float32", [None, 270, 480, 3])
 labels = tf.placeholder("float32", [None, 270, 480, 2])

 # Build the graph
 x = model.Deeplabv3(image)
 print("Deeplabv3 network built!")

 # Printing the shape of the last layer
 print("Shape of the last layer: ", x.get_shape())

 #Loss layer
  l = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=x, labels=labels))

 #Define optimizer
 optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(l)

 print("Running the network...")

 #Initialising global variables
 init = tf.global_variables_initializer()
 sess.run(init)

 # Creating a saver which saves the most recent 1000 checkpoints
 saver = tf.train.Saver(max_to_keep = 1000)

 # Getting the list of training data
 train_X, train_y = getData(data_path)

 # Training
 loss = 0 # Stores the loss after every iteration
 itr = len(train_X)//batch_size # Calculating the no. of epochs
 for i in range(n_epochs):
  for batch in range(itr):
   # Iterate for all images and send based on batch size
   img = train_X[batch*batch_size:(batch+1)*batch_size]
   gt = train_y[batch*batch_size:(batch+1)*batch_size]

   opt = sess.run(optimizer, feed_dict = {image: img, labels:gt})
   loss = sess.run(l, feed_dict = {image: img, labels:gt})
   if batch%20==0:
    print "Epoch: ", str(i), " [", batch, "/", str(itr), "]", " loss: ", loss

  # Saving the model
  if not os.path.exists("trained_models"):
   print("Creating trained models directory...")
   os.makedirs("trained_models")

  if i%10==0:
   os.makedirs("trained_models/"+str(i))
   saver.save(sess, 'trained_models/' + str(i) + '/' + saved_model_name, global_step=i)
   test.test(x, i, image)

 #Closing the session
 sess.close()

train()

