from .layer import *
from .net import Net
from .io import *
from .util import *

# compatible with onnxruntime
InferenceSession = read_net

# planer array library
pal = None


try: import cupy as cp
except: cp = None
try: import numpy as np
except: np == None
try: import numexpr as ep
except: ep = None

print('numpy:[%s] numexpr:[%s] cupy:[%s] '%tuple(
	[('installed', '--')[i is None] for i in (np, ep, cp)]))

def core(obj, silent=False):
	global pal; pal = obj
	from . import util, layer, net, io
	util.np = layer.np = net.np = io.np = obj

	try: import numexpr as ep
	except: ep = None

	layer.ep = ep if obj.__name__ == 'numpy' else None
	if obj.__name__=='numpy' and ep is None:
		print('numexpr is not installed, optional but recommended.')
	pal.cpu = pal.asnumpy if 'asnumpy' in dir(pal) else pal.asarray
	if not silent: print('\nuser switch engine:', obj.__name__)   
	return pal 

core(np, True)

def asnumpy(arr, **key): return pal.cpu(arr, **key)

def asarray(arr, **key): return pal.asarray(arr, **key)
