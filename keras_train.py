import os
import numpy as np
import model
from keras import models
from skimage.io import imread
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

os.system('rm -rf trained_models_keras; mkdir trained_models_keras')

saved_model_name = "trained_models_keras/icnet_model"
h = 256
w = 256
n_samples = 476
n_classes = 2
lr = 1e-3
n_epochs = 20
n_batch = 1
data_path = '/home/ubuntu/Datasets/ISLES2018/train.txt'

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
 X_train, y_train = getData()
 m = model.build(w, h, 2)
 #m = models.Model(inputs=inpt, outputs=x)

 checkpoint = ModelCheckpoint(saved_model_name+".h5")

 m.compile(optimizer = SGD(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])


 m.fit(np.array(X_train), np.array(y_train), epochs=n_epochs,
    batch_size=n_batch, callbacks=[checkpoint])

train()
