from .layer import *
from .net import Net
from .io import *
from .util import *

# compatible with onnxruntime
InferenceSession = read_net

# planer array library
pal = None
import numpy as np
try: import cupy as cp
except: cp = None
try: import numexpr as ep
except: ep = None
try: import cupy.cudnn as dnn
except: dnn = None

print('numpy:[%s] numexpr:[%s] cupy:[%s] cudnn:[%s] '%tuple(
	[('installed', '--')[i is None] for i in (np, ep, cp, dnn)]))

def core(obj, silent=False):
	global pal; pal = obj
	from . import util, layer, net, io
	util.np = layer.np = net.np = io.np = obj
	#try: import numexpr as ep
	#except: ep = None

	layer.ep = ep if obj.__name__ == 'numpy' else None
	layer.dnn = util.dnn = dnn if obj.__name__ == 'cupy' else None

	if obj.__name__=='numpy' and ep is None:
		print('numexpr is not installed, optional but recommended.')
	if obj.__name__=='cupy' and dnn is None:
		print('cudnn is not installed, optional but recommended.')
	pal.asnumpy = pal.asnumpy if 'asnumpy' in dir(pal) else pal.asarray
	if not silent: print('\nuser switch engine:', obj.__name__)   
	return pal 

core(np, True)

def asnumpy(arr, **key): return pal.asnumpy(arr, **key)

def asarray(arr, **key): return pal.asarray(arr, **key)
