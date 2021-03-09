from .layer import layer_map as key
from time import time
from .util import np

class Net:
    def __init__(self):
        self.body, self.flow = [], []
        self.life, self.timer = {}, {}

    def load_json(self, body, flow):
        self.body, self.flow, self.life = [], [], {}
        for i in body:
            para = i[2] or []
            self.body.append((i[0], key[i[1]](*para)))
        for i in range(len(flow)):
            keys = flow[i][0]
            if isinstance(keys, str): keys = [keys]
            for j in keys: self.life[j] = i
            
        self.layer, self.flow = body, flow

    def forward(self, x):
        dic = dict(self.body)
        rst = {self.flow[0][0]: x, 'None': None}
        for i in range(len(self.flow)):
            x, ls, y = self.flow[i]
            if not isinstance(ls, list): ls = [ls]
            for l in ls:
                out = x if l == ls[0] else y
                if not isinstance(out, str):
                    p = [rst[i] for i in out]
                else: p = rst[out]
                xs = x if isinstance(x, list) else [x]
                for k in xs: # release wasted obj
                    if self.life[k]<=i: del rst[k]
                obj = dic[l]
                start = time()
                rst[y] = obj(p)
                cost = time()-start
                if not obj.name in self.timer:
                    self.timer[obj.name] = 0
                self.timer[obj.name] += cost
        return rst[y]

    def layer2code(self, style='list'):
        body = []
        if style == 'list':
            body = ['self.layer = [']
            for i in self.body:
                body.append('\t("%s", %s, %s),' % (i[0],
                    i[1].__class__.__name__, i[1].para()))
            body.append(']')
        if style == 'self':
            body = []
            for i in self.body:
                body.append('self.%s = %s%s' % (i[0],
                    i[1].__class__.__name__, i[1].para() or ()))
        return '\n'.join(body)

    def layer2json(self):
        body = []
        invk = dict(zip(key.values(), key.keys()))
        for i in self.body:
            body.append((i[0], invk[i[1].__class__], i[1].para()))
        return body

    def flw2code(self, style='list'):
        body = []
        if style == 'list':
            for x, ls, y in self.cmds:
                for l in ls:
                    out = x if l == ls[0] else y
                    if isinstance(out, list):
                        out = str(out).replace("'", '')
                    body.append("%s = dic['%s'](%s)" % (y, l, out))
                body.append('')
        if style == 'self':
            for x, ls, y in self.cmds:
                for l in ls:
                    out = x if l == ls[0] else y
                    if isinstance(out, list):
                        out = str(out).replace("'", '')
                    body.append('%s = self.%s(%s)' % (y, l, out))
                body.append('')
        return '\n'.join(body)

    def load_weights(self, data):
        s = 0
        for i in self.body:
            s += i[1].load(data[s:])


    def show(self, info=True):
        from .plot import plot_net
        plot_net(self.layer, self.flow, info).show()

    def __call__(self, x):
        return self.forward(x)


if __name__ == '__main__':
    pass
