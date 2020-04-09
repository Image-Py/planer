import sys
sys.path.append('../../')
import cupy as cp
from time import time
from skimage import io, transform
from matplotlib import pyplot as plt
import cupy
import numpy
import planer
from planer import read_onnx, resize

# get planer array library, numpy
# pal = planer.core(numpy)

# get planer array library, cupy
pal = planer.core(numpy)

def normalize(img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = img.astype('float32')
    img.reshape((-1, 3))[:] -= pal.array(mean) * 255
    img.reshape((-1, 3))[:] /= pal.array(variance) * 255
    return img

# size to be 32x
def makesize32(img):
    h, w = img.shape[-2:]
    w = w // 32 * 32
    h = h // 32 * 32
    return resize(img, (h, w))


img = io.imread('test.jpg')[:, :, ::-1]
img_ = pal.array(img)
x = normalize(img_).transpose(2, 0, 1)[None, :, :, :].copy()
x = makesize32(x)

# 2 files needed, craft.txt, craft.npy
net = read_onnx('craft')
print('load done!')

y = net(x)
'''
net.timer = {}
start = time()
print('start timing!')
for i in range(10):
    y = net(x)
print('planer craft time (x10):', time()-start)
runtime = (time()-start)/10

for k in net.timer:
    print(k, net.timer[k])

y = pal.cpu(y)

plt.subplot(131)
plt.imshow(img[:, :, ::-1].astype('uint8'))
# text map
plt.subplot(132)
plt.imshow(y[0, 0, :, :])
plt.title('text map')
# link map
plt.subplot(133)
plt.imshow(y[0, 1, :, :])
plt.title('link map:%.3fs' % (runtime))

plt.show()
'''
