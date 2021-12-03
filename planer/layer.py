from .util import maxpool, upsample, avgpool, np
from .util import conv_for, conv_stride, conv_dnn
ep, dnn = None, None # numexpr is help for numpy backend
import numpy

def wrap(f, layername='layer'):
    class Layer:
        name = layername
        def __init__(self, **key): self.key = key
        def para(self): return self.key
        def forward(self, *x): return f(*x, **self.key)
        def __call__(self, *x): return self.forward(*x)
    return Layer

def Dense(x, K, B, shp=None):
    y = x.dot(K.T)
    y += B.reshape((1, -1))
    return y

def Conv2d(x, K, B=None, group=1, strides=(1,1), dilations=(1,1), pads=(0,0,0,0)):
    if np is numpy: out = conv_for(x, K, group, pads, strides, dilations)
    elif not dnn is None: return conv_dnn(x, K, B, group, pads, strides, dilations)
    else: out = conv_stride(x, K, group, pads, strides, dilations)
    return out if B is None else np.add(out, B.reshape(1, -1, 1, 1), out=out)

def ConvTranspose2d(x, K, B=None, strides=[2,2], dilations=[1,1], pads=[0,0,0,0], output_padding=[0,0], group=1):
    (n, c, h, w), (s1, s2), (d1, d2), (H, W) = x.shape, strides, dilations, K.shape[2:]
    low_h, high_h = ((H-1)*d1- pads[0]), ((H-1)*d1-pads[2]+output_padding[0])
    low_w, high_w = ((W-1)*d2- pads[1]), ((W-1)*d2-pads[3]+output_padding[1])
    buf = np.zeros((n, c, (h-1)*s1+low_h+high_h+1, (w-1)*s2+low_w+high_w+1), dtype=x.dtype)
    buf[:,:,low_h:buf.shape[2]-high_h:s1,low_w:buf.shape[3]-high_w:s2] = x
    return Conv2d(buf, K.transpose(1,0,2,3)[:,:,::-1,::-1], B, strides=[1,1], dilations=dilations, group=group)

def ReLU(x): 
    if ep: return ep.evaluate('x * (x > 0)')
    return np.multiply(x, x>0, out=x)

def LeakyReLU(x, alpha=0.2):
    a, b = np.float32(alpha), np.float32(1-alpha)
    if ep: return ep.evaluate('x*((x>0)*b+a)')
    y = (x>0) * b; y += a; y *= x; return y

def Flatten(x): return x.reshape((x.shape[0], -1))

def Sigmoid(x):
    if ep: return ep.evaluate('1/(1+exp(-x))')
    x = -x; np.exp(x, out=x); x += 1
    return np.divide(1, x, out=x)

def Softmax(x, axis=-1):
    eX = np.exp((x.T - np.max(x, axis=self.axis)).T)
    return (eX.T / eX.sum(axis=self.axis)).T

def Maxpool(x, w=(2,2), pads=(0,0,0,0), strides=(2,2)):
    return maxpool(x, w, pads, strides)

def Avgpool(x, w=(2,2), pads=(0,0,0,0), strides=(2,2)):
    return avgpool(x, w, pads, strides)

def GlobalAveragePool(x):
    return x.mean(axis=(-2, -1), keepdims=True)

def UpSample(x, k, mode='nearest'):
    if k.size == 0: k = size[-2:] // np.array(x.shape[-2:])
    return upsample(x, k[-2:].astype(int).tolist(), mode)

def Resize(x, roi, k, size=None, mode='nearest', 
    coordinate_transformation_mode='half_pixel', nearest_mode='round_prefer_floor'):
    if k.size == 0: k = size[-2:] // np.array(x.shape[-2:])
    return upsample(x, k[-2:].astype(int).tolist(), mode, 
        coordinate_transformation_mode, nearest_mode)

def Concatenate(*xs, axis=0):
    return np.concatenate(xs, axis=axis)

def Add(x1, x2): 
    if ep: return ep.evaluate('x1 + x2')
    return x1 + x2

def Sub(x1, x2): 
    if ep: return ep.evaluate('x1 - x2')
    return x1 - x2

def Pow(x, p): return np.power(x, p)
    
def Div(x1, x2): return x1 / x2

def ReduceSum(x, axis=-1, keepdims=False):
    return x.sum(axis=tuple(axis), keepdims=keepdims)

def ReduceMean(x, axis=-1, keepdims=False):
    return x.mean(axis=tuple(axis), keepdims=keepdims)

def BatchNorm(x, K, B):
    if ep: return ep.evaluate('x * K + B')
    x = x * K; x += B; return x

def Unsqueeze(x, axis=None): 
    axis = np.array(axis).tolist()
    return np.expand_dims(x, tuple(axis))

def Mul(x1, x2): 
    if ep: return ep.evaluate('x1 * x2')
    return x1 * x2

def Const(value=0, dtype='float32'): 
    if isinstance(value, list):
        return np.array(value, dtype=dtype)
    return value

def LogSoftmax(x, axis=-1):
    y = x - np.max(x, axis=axis, keepdims=True)
    eX = np.sum(np.exp(y), axis=axis, keepdims=True)
    y -= np.log(eX); return y

def Shape(x): return np.array(x.shape)

def Gather(x, idx, axis=0): return np.take(x, idx, axis=axis)

def Reshape(x, shp): 
    for i in range(len(shp)): 
        shp[i] = shp[i] or x.shape[i]
    return x.reshape(shp.tolist())

def Transpose(x, axis): return x.transpose(axis)

def ConstantofShape(x, value=0, dtype='float32'):
    return np.full(x.ravel().tolist(), value, dtype=dtype)

def Split(x, split=None, axis=0):
    seg = np.cumsum(np.array(split)).tolist()
    return np.split(x[:seg[-1]], seg[:-1], axis)

def Tanh(x): 
    if ep: return ep.evaluate('tanh(x)')
    return np.tanh(x)

def Exp(x): 
    if ep: return ep.evaluate('exp(x)')
    return np.exp(x)

def Log(x): 
    if ep: return ep.evaluate('log(x)')
    return np.log(x)

def Slice(x, start, end, axis=None, step=None):
    if step is None: step = np.ones(len(start), dtype=np.uint32)
    if axis is None: axis = np.arange(len(start))
    seas = [start, end, axis, step]
    start, end, axis, step = [i.tolist() for i in seas]
    slis = [slice(None,None,None)] * x.ndim
    for s, e, a, st in zip(start, end, axis, step):
        slis[a] = slice(s, e, st)
    return x[tuple(slis)]

def Expand(x, shp):
    ones = np.ones(shp.tolist(), dtype=x.dtype)
    return ones * x

def Cast(x, dtype='flaot32'): return x.astype(dtype)

def Range(start, end, delta): 
    return np.arange(int(start), int(end), int(delta))

def Equal(x1, x2): return np.equal(x1, x2)

def Where(msk, x1, x2): return np.where(msk, x1, x2)

def Scatternd(data, indices, updates):
    data = data.copy()
    for i in range(len(indices[0])):
        data[tuple(indices[0,i])] = updates[0,i]
    return data

def InstanceNormalization(x, s, bias, epsilon=1e-5):
    axis = tuple(range(2, x.ndim))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = x - mean; var **= 2; 
    var = np.mean(var, axis=axis, keepdims=True)
    shapes = (-1,) + (1,) * (x.ndim - 2)
    s.shape = bias.shape = shapes
    var = (var + epsilon) ** 0.5
    x *= s/var; x += bias - s*mean/var
    return x

def Pad(x, pads, constant_value=0, mode='constant'):
    pads = pads.reshape(2,-1).T.tolist()
    para = {'mode': mode}; 
    if mode=='constant': para['constant_values'] = constant_value
    return np.pad(x, pads, **para)

def Clip(x, min=0, max=1): 
    return np.clip(x, min, max, out=x)

def Return(*x): return x

layer_map = {'dense': Dense, 'conv': Conv2d, 'relu': ReLU, 
             'leakyrelu': LeakyReLU, 'batchnorm': BatchNorm,
             'flatten': Flatten, 'sigmoid': Sigmoid, 'softmax': Softmax, 
             'maxpool': Maxpool, 'avgpool': Avgpool, 'const': Const,
             'upsample': UpSample, 'concat': Concatenate, 'add': Add, 
             'resize': Resize, 'pad': Pad, 'convtranspose':ConvTranspose2d,
             'sub': Sub, 'reducemean': ReduceMean, 'exp': Exp, 'log': Log,
             'mul': Mul, 'gap': GlobalAveragePool, 'pow':Pow,
             'reducesum':ReduceSum, 'div':Div, 'unsqueeze':Unsqueeze, 
             'shape': Shape, 'gather':Gather, 'reshape':Reshape,
             'split':Split, 'tanh':Tanh, 'constantofshape':ConstantofShape,
             'slice':Slice, 'expand':Expand, 'cast':Cast, 'range':Range, 
             'equal':Equal, 'where':Where, 'scatternd':Scatternd,
             'instancenormalization':InstanceNormalization, 'clip':Clip,
             'transpose':Transpose, 'logsoftmax':LogSoftmax, 'return':Return}

if __name__ == "__main__": pass
