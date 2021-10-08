from .util import conv, maxpool, upsample, avgpool, np
import numpy; ep = None # numexpr

def select(x): return (numpy, np)[isinstance(x, np.ndarray)]

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

def Conv2d(x, K, B, shp=None, group=(1,1), strides=(1,1), dilation=(1,1), pads=(1,1)):
    out = conv(x, K, group, pads, strides, dilation)
    out += B.reshape(1, -1, 1, 1)
    return out

def ReLU(x): 
    if ep: return ep.evaluate('x * (x > 0)')
    return np.multiply(x, x>0, out=x)

def LeakyReLU(x, alpha=0.2):
    xalpha = x * alpha
    x2 = np.array([x, xalpha])
    x2 = x2.reshape((2,-1)).max(axis=0)
    return x2.reshape(x.shape)

def Flatten(x): return x.reshape((x.shape[0], -1))

def Sigmoid(x):
    if ep: return ep.evaluate('1/(1+exp(-x))')
    x *= -1; np.exp(x, out=x); x += 1
    return np.divide(1, x, out=x)

def Softmax(x, axis=-1):
    eX = np.exp((x.T - np.max(x, axis=self.axis)).T)
    return (eX.T / eX.sum(axis=self.axis)).T

def Maxpool(x, w=(2,2), pads=(0,0), strides=(2,2)):
    return maxpool(x, w, pads, strides)

def Avgpool(x, w=(2,2), pads=(0,0), strides=(2,2)):
    return avgpool(x, w, pads, strides)

def GlobalAveragePool(x):
    return x.mean(axis=(-2, -1), keepdims=True)

def UpSample(x, k, mode='nearest'):
    return upsample(x, k[-2:], mode)

def Concatenate(*xs, axis=0):
    return select(xs[0]).concatenate(xs, axis=axis)

def Add(x1, x2): 
    if ep: return ep.evaluate('x1 + x2')
    return x1 + x2

def Pow(x, p): return np.power(x, p)
    
def Div(x1, x2): return x1 / x2

def ReduceSum(x, axis, keep_dim):
    if keep_dim:
        return np.expand_dims(x.sum(axis=axis), axis)
    else: return x.sum(axis=axis)

def BatchNorm(x, K, B):
    if ep: return ep.evaluate('x * K + B')
    x = x * K; x += B; return x

def Unsqueeze(x, dim): return select(x).expand_dims(x, dim)

def Mul(x1, x2): 
    if ep: return ep.evaluate('x1 * x2')
    return x1 * x2

def Const(value=0, dtype='float32'): return value

def Return(*x): return x

def LogSoftmax(x, axis=-1):
    y = x - np.max(x, axis=axis, keepdims=True)
    eX = np.sum(np.exp(y), axis=axis, keepdims=True)
    y -= np.log(eX); return y

def Shape(x): return numpy.array(x.shape, dtype=np.int64)

def Gather(x, idx, axis=0): return select(x).take(x, idx, axis=axis)

def Reshape(x, shp): return x.reshape(shp)

def Transpose(x, axis): return x.transpose(axis)

def ConstantofShape(x, value=0, dtype='float32'):
    u = (np, numpy)['int' in dtype]
    return u.full(x.ravel(), value, dtype=dtype)

def Split(x, indices, axis):
    seg = numpy.cumsum(indices)
    return select(x).split(x[:seg[-1]], seg[:-1], axis)

def Tanh(x): 
    if ep: return ep.evaluate('tanh(x)')
    return np.tanh(x)

def Slice(x, start, end, axis, step=None):
    if step is None: step = [1]*len(start)
    seas = [start, end, axis, step]
    start, end, axis, step = [i for i in seas]
    slis = [slice(None,None,None)] * x.ndim
    for s, e, a, st in zip(start, end, axis, step):
        slis[a] = slice(s, e, st)
    return x[tuple(slis)]

def Expand(x, shp):
    ones = np.ones(shp, dtype=x.dtype)
    return ones * x

def Cast(x, dtype='flaot32'): return x.astype(dtype)

def Range(start, end, delta): return np.arange(start, end, delta)

def Equal(x1, x2): return select(x1).equal(x1, x2)

def Where(msk, x1, x2): return select(msk).where(msk, x1, x2)

def Scatternd(data, indices, updates):
    data = data.copy()
    for i in range(len(indices[0])):
        data[tuple(indices[0,i])] = updates[0,i]
    return data

layer_map = {'dense': Dense, 'conv': Conv2d, 'relu': ReLU, 
             'leakyrelu': LeakyReLU, 'batchnorm': BatchNorm,
             'flatten': Flatten, 'sigmoid': Sigmoid, 'softmax': Softmax, 
             'maxpool': Maxpool, 'avgpool': Avgpool, 'const': Const,
             'upsample': UpSample, 'concat': Concatenate, 'add': Add, 
             'mul': Mul, 'gap': GlobalAveragePool, 'pow':Pow,
             'reducesum':ReduceSum, 'div':Div, 'unsqueeze':Unsqueeze, 
             'shape': Shape, 'gather':Gather, 'reshape':Reshape,
             'split':Split, 'tanh':Tanh, 'constantofshape':ConstantofShape,
             'slice':Slice, 'expand':Expand, 'cast':Cast, 'range':Range, 
             'equal':Equal, 'where':Where, 'scatternd':Scatternd,
             'transpose':Transpose, 'logsoftmax':LogSoftmax, 'return':Return}

if __name__ == "__main__": pass
