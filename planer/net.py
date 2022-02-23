from .layer import wrap, layer_map as key
import numpy, time
from .util import np, clear_buf

class Net:
    def __init__(self):
        self.weights, self.body, self.flow = [], [], []
        self.life, self.timer = {}, {}

    def load_json(self, inputs, inits, body, flow, debug=False):
        self.body, self.flow, self.life = [], [], {}
        for i in body:
            para = i[2]
            if debug: print(i)
            self.body.append((i[0], wrap(key[i[1]], i[1])(**para)))
        for i in range(len(flow)):
            keys = flow[i][0]
            if isinstance(keys, str): keys = [keys]
            for j in keys: self.life[j] = i
        for i in inits:
            self.weights.append(np.zeros(i[1], dtype=i[2]))

        self.input, self.inits = inputs, [i[0] for i in inits]
        self.layer, self.flow = body, flow

    def half(self):
        for i in range(len(self.weights)):
            if self.weights[i].dtype == np.float32:
                self.weights[i] = self.weights[i].astype('float16')

    def info(self, obj):
        if isinstance(obj, list):
            return [self.info(i) for i in obj]
        if hasattr(obj, 'shape'): return obj.shape
        return obj

    def forward(self, *x, debug=False):
        dic = dict(self.body)
        rst = {'None': None}
        for k, v in zip(self.inits, self.weights): rst[k] = v

        for k, v in zip(self.input, x): rst[k] = v
        for i in range(len(self.flow)):
            x, ls, y = self.flow[i]
            if not isinstance(ls, list): ls = [ls]
            for l in ls:
                out = x if l == ls[0] else y
                if not isinstance(out, str):
                    p = [rst.get(i) for i in out]
                else: p = [rst[out]]
                xs = x if isinstance(x, list) else [x]
                for k in set(xs): # release wasted obj
                    if k in rst and self.life[k]<=i: del rst[k]
                obj = dic[l]
                start = time.time()
                if debug:
                    print(l, obj.name, ':', obj.para())
                    outp = out #[(i, 'Weights')[i in self.inits] for i in out]
                    print('\t--> ', outp, ':', self.info(p))
                if isinstance(y, str): rst[y] = obj(*p)
                else:
                    for k, v in zip(y, obj(*p)): rst[k] = v
                if debug: 
                    for k in (y, [y])[isinstance(y, str)]:
                        print('\t<-- ',  k, ':', self.info(rst[k]))
                # np.cuda.runtime.deviceSynchronize()
                cost = time.time()-start
                if not obj.name in self.timer:
                    self.timer[obj.name] = 0
                self.timer[obj.name] += cost
        clear_buf()
        return rst[y]

    def timeit(self, status='start'):
        if status == 'start': self.timer = {}
        if status == 'end':
            for i in self.timer: print(i, self.timer[i])

    def run(self, output=None, input={}):
        rst = self(input) # compatible with onnxruntime
        return rst if isinstance(rst, tuple) else (rst,)

    def load_weights(self, data):
        s, data = 0, data.view(dtype=np.uint8)
        for i in range(len(self.weights)):
            buf = self.weights[i].ravel().view(dtype=np.uint8)
            buf[:] = data[s:s+buf.size]
            s += buf.size
                
    def show(self):
        from .plot import plot_net
        plot_net(self.input, self.inits, self.layer, self.flow)

    def __call__(self, *x, **key):
        if type(x[0]) is dict: x = [x[0][i] for i in self.input]
        tp = [isinstance(i, numpy.ndarray) for i in x]
        need = sum(tp)>0 and not numpy is np
        if need: x = [np.asarray(i) if b else i for i,b in zip(x, tp)]
        rst = self.forward(*x, **key)
        if need: rst = tuple([i.get() for i in rst])
        return rst[0] if len(rst)==1 else rst


if __name__ == '__main__':
    pass
