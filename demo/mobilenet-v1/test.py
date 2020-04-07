import numpy as np
from npcnn import read_onnx
from time import time

net = read_onnx('mobile')

x = np.random.randn(1, 3, 224, 224).astype('float32')

net(x)

start = time()

for i in range(1):
    net(x)
print('npcnn mobilenet time:', time()-start)
