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
	global np; np = obj
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
	np.asnumpy = obj.asnumpy if 'asnumpy' in dir(obj) else obj.asarray
	if not silent: print('\nuser switch engine:', obj.__name__)   
	return np 

core(np, True)

def asnumpy(arr, **key): return np.asnumpy(arr, **key)

def asarray(arr, **key): return np.asarray(arr, **key)

# ========== planer zoo ==========
import inspect
import urllib.request

root = os.path.expandvars('$HOME')+'/.planer_zoo'
if not os.path.exists(root): os.mkdir(root)

def progress(i, n, bar=[None]):
    from tqdm import tqdm
    if bar[0] is None:
        bar[0] = tqdm()
    bar[0].total = n
    bar[0].update(i-bar[0].n)
    if n==i: bar[0] = None
   
def download(url, path, info=print, progress=progress):
    info('download from %s'%url)
    f, rst = urllib.request.urlretrieve(url, path,
        lambda a,b,c: progress(int(100.0 * a * b/c), 100))

def source(root, lst):
    for i in lst:
        if len(i)==3: i.insert(2, False)
        i[2] = os.path.exists(root + '/' + i[0])
    return lst

def list_source(root, lst):
    print('%-20s%-10s%-10s\n'%('file name','required', 'installed')+'-'*40)
    for i in source(root, lst):print('%-20s%-10s%-10s'%(tuple(i[:3])))

def downloads(root, lst, names='required', force=False, info=print, progress=progress):
    source(root, lst)
    if names=='all': lst = [i for i in lst]
    elif names=='required': lst = [i for i in lst if i[1]]
    else:
        if isinstance(names, str): names = [names]
        lst = [i for i in lst if i[0] in names]
    if not force: lst = [i for i in lst if not i[2]]
    # name = model.__name__.replace('planer_zoo.', '')
    if not os.path.exists(root): os.makedirs(root)
    for name, a, b, url in lst:
        download(url, root+'/'+name, info, progress)

# parse source from a markdown file
def get_source(path):
    with open(path) as f: cont = f.read().split('\n')
    status, files = False, []
    for i in range(len(cont)):
        if '|File|' in cont[i].replace(' ',''): break
    for i in range(i, len(cont)):
        if not '|' in cont[i]: break
        if not '](' in cont[i]: continue
        nameurl = cont[i].split('|')[1]
        req = cont[i].split('|')[2].strip()!=''
        name, url = nameurl.split('](')
        name = name.split('[')[1]
        url = url.split(')')[0]
        files.append([name, req, url])
    return files

def Model(model, auto=True):
    name = model.__name__.replace('planer_zoo.', '')
    md = model.__file__.replace('__init__.py', 'readme.py')[:-2]+'md'
    mroot = root +'/' +  '/'.join(name.split('.'))
    if hasattr(model, 'source'): 
        lst = [list(i) for i in model.source]
        model.source = lambda m=mroot: source(m, lst)
    else: model.source = lambda m=mroot: source(m, get_source(md))
    model.root, oroot = mroot, model.root
    ms = [getattr(model, i) for i in dir(model)]
    for m in set([inspect.getmodule(i) for i in ms]):
        if hasattr(m, 'root') and m.root == oroot: m.root = mroot
    model.list_source = lambda root=mroot, lst=model.source(): list_source(root, lst)
    model.download = lambda name='required', force=False, info=print, \
    	progress=progress, m=mroot: downloads(
    		m, model.source(), name, force, info, progress)
    if auto: [model.download(), model.load()]
    return model