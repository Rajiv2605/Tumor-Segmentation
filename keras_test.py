import numpy as np
import model
from skimage.io import imread
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD

h = 256
w = 256

def dice(pred, label):
    pred = np.argmax(pred[0].transpose(), axis=0)
    print np.unique(pred)
    pred = pred.transpose()
    dice_temp = np.float(np.sum(pred[label==1] == 1))*2.0 / (np.float(np.sum(label == 1) + np.sum(pred == 1)));
    return dice_temp


#u = UNet()
m = model.build(w, h, 2)
#m = u.create_model((w,h,3), 2)
m.load_weights("trained_models_keras/icnet_model.h5")

m.compile(optimizer = SGD(lr = 1e-4), loss = 'binary_crossentropy')

for layer in m.layers:
    weights = layer.get_weights() # list of numpy arrays

for w in weights:
    print layer.name, w
# IMG = []
# IMG.append(imread(sample_input))


# output = m.predict(np.array(IMG))
# output = np.argmax(output[0].transpose(), axis=0)

d = 0
idx = 0
imgs = []

with open('/home/ubuntu/Datasets/ISLES2018/train.txt', 'r') as f:
    for line in f:
      idx += 1
        input_image_file, ground_truth_file = line.split();
        image_ori = imread(input_image_file);
        gt = imread(ground_truth_file)
        imgs.append(image_ori)
        d += dice(m.predict(np.array(imgs)), gt)
        imgs[:] = []
        if idx==10:
         break

print "Dice: ", d/10
