from .util import conv, maxpool, upsample, avgpool, np
import numpy as cpu

def ls(x): return (x, [x,x])[type(x) in {int, float}]

def select(x): return (cpu, np)[isinstance(x, np.ndarray)]

class Layer:
    name = 'layer'

    def __init__(self): pass

    def forward(self, x): pass

    def backward(self, grad_y): pass

    def para(self): return None

    def load(self, buf): return 0

    def __call__(self, *x):
        return self.forward(*x)


class Dense(Layer):
    name = 'dense'

    def __init__(self, c, n): pass
        #self.K = np.zeros((n, c), dtype=np.float32)
        #self.bias = np.zeros(n, dtype=np.float32)

    # def para(self): return self.K.shape

    def forward(self, x, K, B):
        y = x.dot(K.T)
        y += B.reshape((1, -1))
        return y

class Conv2d(Layer):
    name = 'conv'

    def __init__(self, c, n, w, g=(1,1), s=(1,1), d=(1,1), p=(1,1)):
        """
        c: in_channels
        n: out_channels
        w: kernel_size
        g: groups
        s: stride
        d: dilation
        p: padding
        """
        self.n, self.c, self.w = n, c, ls(w)
        self.g, self.s, self.d = g, ls(s), ls(d)
        self.p = ls(p)

        #self.K = np.zeros((n, c, *ls(w)), dtype=np.float32)
        #self.bias = np.zeros(n, dtype=np.float32)

    def para(self):
        return self.n, self.c, self.w, self.s, self.d, self.p

    def forward(self, x, K, B):
        out = conv(x, K, self.g, self.p, self.s, self.d)
        out += B.reshape(1, -1, 1, 1)
        return out


class ReLU(Layer):
    name = 'relu'

    def __init__(self): pass

    def forward(self, x):
        return np.multiply(x, x>0, out=x)

class LeakyReLU(Layer):
    name = 'leakyrelu'

    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def forward(self, x):
        xalpha = x * self.alpha
        x2 = np.array([x, xalpha])
        x2 = x2.reshape((2,-1)).max(axis=0)
        return x2.reshape(x.shape)

class Flatten(Layer):
    name = 'flatten'

    def __init__(self): pass

    def forward(self, x):
        return x.reshape((x.shape[0], -1))


class Sigmoid(Layer):
    name = 'sigmoid'

    def __init__(self): pass

    def forward(self, x):
        x *= -1; np.exp(x, out=x); x += 1
        return np.divide(1, x, out=x)


class Softmax(Layer):
    name = 'softmax'

    def __init__(self, axis=-1):
        self.axis = axis

    def forward(self, x):
        eX = np.exp((x.T - np.max(x, axis=self.axis)).T)
        return (eX.T / eX.sum(axis=self.axis)).T


class Maxpool(Layer):
    name = 'maxpool'

    def __init__(self, w=(2,2), pad=(0,0), stride=(2,2)):
        self.w, self.pad, self.stride = w, pad, stride

    def para(self): return (self.w, self.pad, self.stride)

    def forward(self, x):
        return maxpool(x, self.w, self.pad, self.stride)


class Avgpool(Layer):
    name = 'avgpool'

    def __init__(self, w=(2,2), pad=(0,0), stride=(2,2)):
        self.w, self.pad, self.stride = w, pad, stride

    def para(self): return (self.w, self.stride)

    def forward(self, x):
        return avgpool(x, self.w, self.pad, self.stride)


class GlobalAveragePool(Layer):
    name = 'gap'
    def __init__(self): pass

    def forward(self, x):
        return x.mean(axis=(-2, -1), keepdims=True)


class UpSample(Layer):
    name = 'upsample'

    def __init__(self, mode):
        self.mode = mode

    def para(self): return self.mode

    def forward(self, x, k):
        return upsample(x, k[-2:], self.mode)


class Concatenate(Layer):
    name = 'concat'

    def __init__(self, axis): self.axis=axis

    def forward(self, *xs):
        return select(xs[0]).concatenate(xs, axis=self.axis)


class Add(Layer):
    name = 'add'

    def __init__(self): pass

    def forward(self, x1, x2):
        return x1 + x2


class Pow(Layer):
    name = 'pow'
    
    def forward(self, x, p):
        return np.power(x, p)

    
class Div(Layer):
    name = 'div'
    
    def __init__(self): pass
    
    def forward(self, x1, x2):
        return x1 / x2

class ReduceSum(Layer):
    name = 'reducesum'
    
    def __init__(self, axis, keep_dim):
        self.axis = axis
        self.keep_dim = keep_dim
    
    def forward(self, x):
        if self.keep_dim:
            return np.expand_dims(x.sum(axis=self.axis), self.axis)
        else:
            return x.sum(axis=self.axis)

class Unsqueeze(Layer):
    name = 'unsqueeze'
    
    def __init__(self, dim):
        self.dim = dim
    
    def forward(self, x):
        return select(x).expand_dims(x, self.dim)


class Mul(Layer):
    name = 'mul'

    def __init__(self): pass

    def forward(self, x1, x2):
        return x1 * x2


class Const(Layer):
    name = 'const'

    def __init__(self, v, tp): 
        self.v = v

    def forward(self): 
        return self.v


class ConstArray(Layer):
    name = 'constarray'
    def __init__(self, shp):
        self.arr = np.zeros(shp, dtype=np.int64)

    def forward(self, x): return self.arr

    def load(self, buf): 
        self.arr.ravel()[:] = buf[:self.arr.size]
        return self.arr.size


class Return(Layer):
    name = 'return'

    def forward(self, *x):
        return x
    

class BatchNorm(Layer):
    name = 'batchnorm'

    def __init__(self):
        self.K = self.B = None

    def forward(self, x, k, b, m, v):
        K, B = self.eval(k, b, m, v)
        x = x * K
        x += B
        return x
        #return self.kv_inv*x + self.kmv_inv_b

    def eval(self, k, b, m, v):
        if not self.K is None: return self.K, self.B
        v_inv = 1/np.sqrt(v + 1e-5)
        kmv_inv_b = -k*m*v_inv + b
        kv_inv = k*v_inv
        kmv_inv_b.shape = kv_inv.shape = (1,-1,1,1)
        self.K, self.B = kv_inv, kmv_inv_b
        return kv_inv, kmv_inv_b

class LogSoftmax(Layer):
    name = 'logsoftmax'

    def __init__(self, axis=-1):
        self.axis = axis
        
    def forward(self, x):
        y = x - np.max(x, axis=self.axis, keepdims=True)
        eX = np.sum(np.exp(y), axis=self.axis, keepdims=True)
        y -= np.log(eX); return y

class Shape(Layer):
    name = 'shape'

    def forward(self, x): return cpu.array(x.shape, dtype=np.int64)

class Gather(Layer):
    name = 'gather'
    def __init__(self, axis=0): 
        self.axis = axis

    def forward(self, x, idx): 
        return select(x).take(x, idx, axis=self.axis)

class Reshape(Layer):
    name = 'reshape'

    def forward(self, x, shp): 
        return x.reshape(shp)

class Transpose(Layer):
    name = 'transpose'
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x): return x.transpose(self.axis)

class ConstantofShape(Layer):
    name = 'constantofshape'
    def __init__(self, v=0, tp='float32'):
        self.v, self.tp = v, tp

    def forward(self, x):
        u = (np, cpu)['int' in self.tp]
        return u.full(x.ravel(), self.v, dtype=self.tp)

class Split(Layer):
    name = 'split'
    def __init__(self, indices, axis):
        self.indices, self.axis = indices, axis
        self.seg = cpu.cumsum(self.indices)

    def forward(self, x):
        return select(x).split(x[:self.seg[-1]], self.seg[:-1], self.axis)

class Tanh(Layer):
    name = 'tanh'

    def forward(self, x):
        return np.tanh(x)

class Slice(Layer):
    name = 'slice'

    def forward(self, x, start, end, axis, step=None):
        if step is None: step = cpu.array([1]*len(start))
        seas = [start, end, axis, step]
        start, end, axis, step = [i for i in seas]
        slis = [slice(None,None,None)] * x.ndim
        for s, e, a, st in zip(start, end, axis, step):
            slis[a] = slice(s, e, st)
        return x[tuple(slis)]

class Expand(Layer):
    name = 'expand'

    def forward(self, x, shp):
        ones = np.ones(shp, dtype=x.dtype)
        return ones * x

class Cast(Layer):
    name = 'cast'
    def __init__(self, fmt): self.fmt = fmt

    def forward(self, x):
        return x.astype(self.fmt)

class Range(Layer):
    name = 'range'

    def forward(self, start, end, delta):
        return np.arange(start, end, delta)

class Equal(Layer):
    name = 'equal'

    def forward(self, x1, x2):
        return select(x1).equal(x1, x2)

class Where(Layer):
    name = 'where'

    def forward(self, msk, x1, x2):
        return select(msk).where(msk, x1, x2)

class Scatternd(Layer):
    name = 'scatternd'

    def forward(self, data, indices, updates):
        for i in range(len(indices[0])):
            data = data.copy()
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
             'constarray':ConstArray, 'slice':Slice, 'expand':Expand,
             'cast':Cast, 'range':Range, 'equal':Equal, 'where':Where,
             'scatternd':Scatternd,
             'transpose':Transpose, 'logsoftmax':LogSoftmax, 'return':Return}

if __name__ == "__main__":
    pass
