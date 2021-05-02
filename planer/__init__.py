from .layer import *
from .net import Net
from .io import read_net, onnx2planer
from .util import *

# planer array library
pal = None

def core(obj):
	global pal
	pal = obj
	from . import util, layer, net, io
	util.np = layer.np = net.np = io.np = obj
	if 'asnumpy' in dir(pal):
		pal.cpu = pal.asnumpy
	else:
		pal.cpu = pal.asarray
	return pal
	print('\nuser switch engine:', core.__name__)    

def asnumpy(arr, **key): return pal.cpu(arr, **key)

def asarray(arr, **key): return pal.asarray(arr, **key)

try:
    import cupy
    core(cupy)
    print('using cupy engine, gpu powered!')
except:
    import numpy as np
    core(np)
    print('using numpy engine, install cupy would be faster.')