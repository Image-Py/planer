from .util import conv, maxpool, upsample, avgpool, np


class Layer:
    name = 'layer'

    def __init__(self, name):
        self.name = name

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

    def __init__(self, c, n, w, g=1, s=1, d=1):
        """
        c: in_channels
        n: out_channels
        w: kernel_size
        g: groups
        s: stride
        d: dilation
        """
        self.n, self.c, self.w = n, c, w
        self.g, self.s, self.d = g, s, d

        self.K = np.zeros((n, c, w, w), dtype=np.float32)
        self.bias = np.zeros(n, dtype=np.float32)

    def para(self):
        return self.n, self.c, self.w, self.s, self.d

    def forward(self, x):
        out = conv(x, self.K, self.g, (self.s, self.s), (self.d, self.d))
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
        #return np.maximum(x, 0, out=x)
        msk = x>0; x *= msk; return x

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

    def __init__(self, w=2, stride=2):
        self.w = w
        self.stride = stride

    def para(self): return (self.w, self.stride)

    def forward(self, x):
        return maxpool(x, (self.w,) * 2, (self.stride,) * 2)


class Avgpool(Layer):
    name = 'avgpool'

    def __init__(self, w=2, stride=2):
        self.w = w
        self.stride = stride

    def para(self): return (self.w, self.stride)

    def forward(self, x):
        return avgpool(x, (self.w,) * 2, (self.stride,) * 2)


class GlobalAveragePool(Layer):
    name = 'gap'
    def __init__(self): pass

    def forward(self, x):
        return x.mean(axis=(-2, -1))


class UpSample(Layer):
    name = 'upsample'

    def __init__(self, k, mode):
        self.k = k
        self.mode = mode

    def para(self): return (self.k,)

    def forward(self, x):
        return upsample(x, self.k, self.mode)


class Concatenate(Layer):
    name = 'concat'

    def __init__(self): pass

    def forward(self, x):
        return np.concatenate(x, axis=1)


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
        return np.expand_dims(x, self.dim)


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


layer_map = {'dense': Dense, 'conv': Conv2d, 'relu': ReLU, 'leakyrelu': LeakyReLU, 'batchnorm': BatchNorm,
             'flatten': Flatten, 'sigmoid': Sigmoid, 'softmax': Softmax, 'maxpool': Maxpool, 'avgpool': Avgpool, 'const': Const,
             'upsample': UpSample, 'concat': Concatenate, 'add': Add, 'mul': Mul, 'gap': GlobalAveragePool, 'pow':Pow,
             'reducesum':ReduceSum, 'div':Div, 'unsqueeze':Unsqueeze, 'return':Return}

if __name__ == "__main__":
    pass
