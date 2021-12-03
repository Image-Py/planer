from .layer import *
from .net import Net
from .io import *
from .util import *

# compatible with onnxruntime
InferenceSession = read_net

# planer array library
backend = None
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
	global backend; backend = obj
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
	backend.asnumpy = obj.asnumpy if 'asnumpy' in dir(obj) else obj.asarray
	if not silent: print('\nuser switch engine:', obj.__name__)   
	return backend 

core(np, True)

def asnumpy(arr, **key): return backend.asnumpy(arr, **key)

def asarray(arr, **key): return backend.asarray(arr, **key)
