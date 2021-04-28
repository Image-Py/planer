import sys
sys.path.append('../../')
import planer
import numpy as np
from time import time
from skimage.io import imread
from skimage.data import page
from matplotlib import pyplot as plt

dic = {1: '!', 2: '@', 3: '#', 4: '$', 5: '%', 6: '&', 7: '(', 8: ')', 9: '+', 10: '-', 11: ':', 12: ',', 13: '.', 14: "'", 15: '"', 16: '/', 17: '\\', 18: '?', 19: '0', 20: '1', 21: '2', 22: '3', 23: '4', 24: '5', 25: '6', 26: '7', 27: '8', 28: '9', 29: 'a', 30: 'b', 31: 'c', 32: 'd', 33: 'e', 34: 'f', 35: 'g', 36: 'h', 37: 'i', 38: 'j', 39: 'k', 40: 'l', 41: 'm', 42: 'n', 43: 'o', 44: 'p', 45: 'q', 46: 'r', 47: 's', 48: 't', 49: 'u', 50: 'v', 51: 'w', 52: 'x', 53: 'y', 54: 'z'}

def greedy_search(raw, blank=0):
    max_id = raw.argmax(2).ravel()
    msk = max_id[1:] != max_id[:-1]
    max_id = max_id[1:][msk]
    return max_id[max_id!=blank]

# net = planer.onnx2planer('./crnn.onnx')
net = planer.read_net('./crnn-ocr')

x = page()[:40,:150].astype('float32')

w = 48*x.shape[1]//x.shape[0]
x = planer.resize(x, (48, w))
x = (x - 0.5)/(90/255)
x = x[None, None, :, :]

net.timer = {}
start = time()
y = net(x)
print(time()-start)

for k in net.timer:
    print(k, net.timer[k])
    
pred = greedy_search(y)
pred = [dic[i] for i in pred]
text = ''.join(pred)

plt.imshow(x[0,0])
plt.title(text)
plt.show()
