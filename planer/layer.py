from .util import conv, maxpool, upsample, avgpool, np

def ls(x): return (x, [x,x])[type(x) in {int, float}]

class Layer:
    name = 'layer'

    def __init__(self): pass

    def forward(self, x): pass

    def backward(self, grad_y): pass

    def para(self): return None

    def load(self, buf): return 0

    def __call__(self, x):
        return self.forward(x)


class Dense(Layer):
    name = 'dense'

    def __init__(self, c, n):
        self.K = np.zeros((n, c), dtype=np.float32)
        self.bias = np.zeros(n, dtype=np.float32)

    def para(self): return self.K.shape

    def forward(self, x):
        y = x.dot(self.K.T)
        y += self.bias.reshape((1, -1))
        return y

    def load(self, buf):
        sk, sb = self.K.size, self.bias.size
        self.K.ravel()[:] = buf[:sk]
        self.bias.ravel()[:] = buf[sk:sk+sb]
        return sk + sb


class Conv2d(Layer):
    name = 'conv'

    def __init__(self, c, n, w, g=1, s=1, d=1, p=1):
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

        self.K = np.zeros((n, c, *ls(w)), dtype=np.float32)
        self.bias = np.zeros(n, dtype=np.float32)

    def para(self):
        return self.n, self.c, self.w, self.s, self.d, self.p

    def forward(self, x):
        out = conv(x, self.K, self.g, self.p, self.s, self.d)
        out += self.bias.reshape(1, -1, 1, 1)
        return out

    def load(self, buf):
        sk, sb = self.K.size, self.bias.size
        self.K.ravel()[:] = buf[:sk]
        self.bias.ravel()[:] = buf[sk:sk+sb]        
        return sk + sb


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
        return 1/(1 + np.exp(-x))


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

    def __init__(self, k, mode):
        self.k = ls(k)
        self.mode = mode

    def para(self): return (self.k, self.mode)

    def forward(self, x):
        return upsample(x, self.k, self.mode)


class Concatenate(Layer):
    name = 'concat'

    def __init__(self, axis): self.axis=axis

    def forward(self, x):
        return np.concatenate(x, axis=self.axis)


class Add(Layer):
    name = 'add'

    def __init__(self): pass

    def forward(self, x):
        x[0] += x[1]
        return x[0]


class Pow(Layer):
    name = 'pow'
    
    def __init__(self, n): self.n = n
    
    def forward(self, x):
        return np.power(x, self.n)

    
class Div(Layer):
    name = 'div'
    
    def __init__(self): pass
    
    def forward(self, x):
        return x[0]/x[1]

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
        return np.expand_dims(np.asarray(x), self.dim)


class Mul(Layer):
    name = 'mul'

    def __init__(self): pass

    def forward(self, x):
        x[1] *= x[0]
        return x[1]


class Const(Layer):
    name = 'const'

    def __init__(self, v): self.v = v

    def forward(self, x): return self.v


class ConstArray(Layer):
    name = 'constarray'
    def __init__(self, shp):
        self.arr = np.zeros(shp, dtype=np.float32)

    def forward(self, x): return self.arr

    def load(self, buf): 
        self.arr.ravel()[:] = buf[:self.arr.size]
        return self.arr.size


class Return(Layer):
    name = 'return'

    def __init__(self): pass

    def forward(self, x):
        return x
    

class BatchNorm(Layer):
    name = 'batchnorm'

    def __init__(self, c):
        self.c = c
        self.k = np.zeros(c, dtype=np.float32)
        self.b = np.zeros(c, dtype=np.float32)
        self.m = np.zeros(c, dtype=np.float32)
        self.v = np.zeros(c, dtype=np.float32)

    def forward(self, x):
        x = x * self.kv_inv
        x += self.kmv_inv_b
        return x
        #return self.kv_inv*x + self.kmv_inv_b

    def load(self, buf):
        c = self.c
        self.k[:] = buf[0*c:1*c]
        self.b[:] = buf[1*c:2*c]
        self.m[:] = buf[2*c:3*c]
        self.v[:] = buf[3*c:4*c]

        self.v_inv = 1/np.sqrt(self.v + 1e-5)
        self.kmv_inv_b = -self.k*self.m*self.v_inv + self.b
        self.kv_inv = self.k*self.v_inv
        self.kmv_inv_b.shape = self.kv_inv.shape = (1,-1,1,1)
        return self.c * 4

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

    def forward(self, x): return x.shape

class Gather(Layer):
    name = 'gather'
    def __init__(self, idx): self.idx = idx

    def forward(self, x): return x[self.idx]

class Reshape(Layer):
    name = 'reshape'

    def forward(self, x): 
        return x[0].reshape(x[1].tolist())

class Transpose(Layer):
    name = 'transpose'
    def __init__(self, axis):
        self.axis = axis

    def forward(self, x): return x.transpose(self.axis)

class ConstantofShape(Layer):
    name = 'constantofshape'
    def __init__(self, v=0):
        self.v = v

    def forward(self, x):
        return np.full(x.astype('int64').ravel(), self.v, dtype=np.float32)

class Split(Layer):
    name = 'split'
    def __init__(self, indices, axis):
        self.indices, self.axis = indices, axis

    def forward(self, x):
        return np.split(x, self.indices, self.axis)

class Tanh(Layer):
    name = 'tanh'

    def forward(self, x):
        return np.tanh(x)

layer_map = {'dense': Dense, 'conv': Conv2d, 'relu': ReLU, 
             'leakyrelu': LeakyReLU, 'batchnorm': BatchNorm,
             'flatten': Flatten, 'sigmoid': Sigmoid, 'softmax': Softmax, 
             'maxpool': Maxpool, 'avgpool': Avgpool, 'const': Const,
             'upsample': UpSample, 'concat': Concatenate, 'add': Add, 
             'mul': Mul, 'gap': GlobalAveragePool, 'pow':Pow,
             'reducesum':ReduceSum, 'div':Div, 'unsqueeze':Unsqueeze, 
             'shape': Shape, 'gather':Gather, 'reshape':Reshape,
             'split':Split, 'tanh':Tanh, 'constantofshape':ConstantofShape,
             'constarray':ConstArray,
             'transpose':Transpose, 'logsoftmax':LogSoftmax, 'return':Return}

if __name__ == "__main__":
    pass
