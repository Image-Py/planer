import sys
sys.path.append('../../')
import numpy
#import cupy
import planer
from planer import read_net, resize
from imagenet_labels import classes
from skimage import io
from matplotlib import pyplot as plt
from time import time

# get planer array library, numpy
# pal = planer.core(numpy)

# get planer array library, cupy
pal = planer.core(numpy)

img = io.imread('test.jpg')[:,:,:3]
x = (img/255.0).transpose(2, 0, 1)
x = x[None, :, :, :].astype('float32')
x = pal.array(x)
x = resize(x, (224, 224))

net = read_net('resnet18')
print('load done!')
y = net(x)
'''
print('start timing!')
net.timer={}
start = time()
for i in range(10):
    y = net(x)
print('planer resnet18 time (x10):', time()-start)
'''
y = pal.argmax(y, axis=-1)
rst = classes[int(y[0])]

for k in net.timer:
    print(k, net.timer[k])

print('result:', rst)
plt.imshow(img.astype('uint8'))
plt.title(rst)
plt.show()
