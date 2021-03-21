import sys
sys.path.append('../../')
import numpy
import cupy
import planer
from planer import read_net, resize
from skimage.data import coins, gravel
from matplotlib import pyplot as plt
from time import time

# get planer array library, numpy
# get planer array library, cupy

pal = planer.core(cupy)

img = gravel()
img = numpy.concatenate(([img], [img]))
img = img.astype('float32')/255

#img = numpy.zeros((2,1024,1024), 'float32')
img_ = pal.array(img)

x = img_[None,:,:,:]

net = read_net('cellpose')
print('load done!')

y = net(x)
net.timer = {}
print('start timing!')
start = time()
for i in range(1): y = net(x)
print('unet detect time x1:', time()-start)

for k in net.timer:
   print(k, net.timer[k])

y = pal.cpu(y[0])
plt.subplot(131)
plt.imshow(y[0,0], 'gray')
plt.title('flow X')

plt.subplot(132)
plt.imshow(y[0,1], 'gray')
plt.title('flow Y')

plt.subplot(133)
plt.imshow(y[0,2], 'gray')
plt.title('prob')
plt.show()

