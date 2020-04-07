import sys
sys.path.append('../../')
from time import time
from matplotlib import pyplot as plt
from skimage import io
import numpy
import cupy
import planer
from planer import read_onnx, resize

# get planer array library, numpy
# pal = planer.core(numpy)

# get planer array library, cupy
pal = planer.core(cupy)

img = io.imread('comic.png')[:, :, :3]
x = (img/255.0).transpose(2, 0, 1)
x = x[None, :, :, :].astype('float32')
x = pal.array(x)

net = read_onnx('ESRGAN')
print('load done!')

y = net(x)
print('start timing!')
start = time()
y = net(x)
print('planer esrgan time:', time()-start)


y = pal.cpu(y)
plt.subplot(121)
plt.imshow(img.astype('uint8'))
plt.subplot(122)
plt.imshow(y[0, :, :, :].transpose(1, 2, 0))
plt.show()
