import sys
sys.path.append('../../')
import cupy as cp
from time import time
from skimage import io
from matplotlib import pyplot as plt
import numpy
import cupy
import planer
from planer import read_onnx, resize


# get planer array library, numpy
# pal = planer.core(numpy)

# get planer array library, cupy
pal = planer.core(cupy)

# size to be 32x
def makesize32(img):
    h, w = img.shape[-2:]
    w = w // 32 * 32
    h = h // 32 * 32
    return resize(img, (h, w))


def normal(img):
    img = img.astype('float32')
    img.reshape((-1, 3))[:] -= pal.array([104, 117, 123])
    return img

img = io.imread('test.jpg')
img_ = pal.array(img)
x = normal(img_).transpose(2, 0, 1)
x = makesize32(x[None, :, :, :])


# 2 files needed, hed.txt, hed.npy
net = read_onnx('hed')
y = net(x)
net.timer = {}
print('start timing!')
start = time()
for i in range(10):
    y = net(x)
print('planer hed time (x10):', time()-start)
run_time = (time()-start)/10

for k in net.timer:
    print(k, net.timer[k])


y = pal.cpu(y)

plt.subplot(121)
plt.imshow(img)
# edge map
plt.subplot(122)
plt.imshow(y[0, 0, :, :])
plt.title('edge map:%.3fs' % (run_time))

plt.show()
