from .layer import *
from .net import Net
from .io import read_net, onnx2planer
from .util import *

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
