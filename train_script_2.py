import sys
sys.path.insert(0, '/home/ubuntu/mobarak/PSPNET-cudnn5/python')
import caffe
import cv2

_weights = '../resnet101-v2-merge.caffemodel'


caffe.set_mode_gpu();
caffe.set_device(3)
solver_path = 'solver.prototxt'

solver = caffe.get_solver(solver_path)
solver.net.copy_from(_weights);
solver.solve()

