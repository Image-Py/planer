from .layer import *
from .net import Net
from .io import read_net, torch2planer, read_onnx
from .util import conv, maxpool, resize

# planer array library
pal = None

def core(obj): 
	pal = obj
	from . import util, layer, net, io
	util.np = layer.np = net.np = io.np = obj
	if 'asnumpy' in dir(pal):
		pal.cpu = pal.asnumpy
	else:
		pal.cpu = pal.array
	return pal