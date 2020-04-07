import planer
from planer import read_onnx
import numpy as np

pal = planer.core(np)
# the same folder should contain resnet18.txt, resnet18.npy
net = read_onnx('resnet18')
# input should be float32
x = pal.random.randn(1, 3, 224, 224).astype('float32')
y = net(x)
print(y.shape)