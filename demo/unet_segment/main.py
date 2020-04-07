import sys
sys.path.append('../../')
import numpy
import cupy
import planer
from planer import read_onnx, resize
from skimage import io
from matplotlib import pyplot as plt
from time import time

# get planer array library, numpy
# pal = planer.core(numpy)

# get planer array library, cupy
pal = planer.core(cupy)


img = io.imread('test.jpg', 0).astype('float32') / 255.0
img_ = pal.array(img)

x = img_[None, None, :, :]

net = read_onnx('unet')
print('load done!')

y = net(x)
net.timer = {}
print('start timing!')
start = time()
for i in range(10):
    y = net(x)
print('unet detect time x10:', time()-start)

for k in net.timer:
   print(k, net.timer[k])

y = pal.cpu(y)
plt.subplot(121)
plt.imshow(img, 'gray')
# edge map
plt.subplot(122)
plt.imshow(y[0, 0, :, :])
plt.title('unet seg')

plt.show()
