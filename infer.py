import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/davince/Downloads/caffe-segnet-segnet-cleaned/python/')
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()


# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
im = Image.open('/home/davince/SegNet/CamVid/val/0016E5_07959.png')
in_ = np.array(im, dtype=np.float32)
in_ = in_[:,:,::-1]
in_ -= np.array((104.00698793,116.66876762,122.67891434))
in_ = in_.transpose((2,0,1))

# load net
net = caffe.Net('/home/davince/SegNet/Models/segnet_inference.prototxt', '/home/davince/SegNet/Models/Inference/test_weights.caffemodel', caffe.TEST)
# shape for input (data blob is N x C x H x W), set data
net.blobs['data'].reshape(1, *in_.shape)
net.blobs['data'].data[...] = in_
# run net and take argmax for prediction
net.forward()
#out = net.blobs['score'].data[0].argmax(axis=0)
out=net.blobs['prob'].data[0].argmax(axis=0)

plt.imshow(out, cmap='gray');plt.axis('off')
plt.savefig('test.png')
plt.show()
