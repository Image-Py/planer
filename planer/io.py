import json, re, os
import numpy as np
from .net import Net
from time import time
import json, zipfile
from io import BytesIO

def read_net(path):
    net = Net()
    if os.path.exists(path+'.pla'):
        with zipfile.ZipFile(path+'.pla') as f:
            path = os.path.split(path)[1]
            body = json.loads(f.read(path+'.json'))
            lay, flw = body['layers'], body['flow']
            buf = BytesIO(f.read(path+'.npy'))
            weights = np.load(buf)
    else:
        with open(path+'.json') as f:
            body = json.load(f)
            lay, flw = body['layers'], body['flow']
        weights = np.load(path+'.npy')
    net.load_json(lay, flw)
    net.load_weights(weights)
    return net

def onnx2planer(path):
    import onnx, onnx.numpy_helper
    graph = onnx.load(path).graph
    layers, weights, flows, const, values = [], [], [], {}, {}
    for i in graph.initializer: values[i.name] = onnx.numpy_helper.to_array(i)
    for i in graph.node:
        # print(i.name)
        has = [i for i in i.input if i in values]
        no = [i for i in i.input if not i in values]
        if len(no)==1: no = no[0]
        if no != []:
            flows.append([no, [i.name], i.output[0]])
            weights.extend([values[i] for i in has])
        if i.op_type == 'BatchNormalization':
            layers.append([i.name, 'batchnorm', [weights[-1].shape[0]]])
        elif i.op_type == 'Conv':
            attr, w = i.attribute, weights[-2].shape
            g, d, p, s = attr[1].i, attr[0].ints[0], attr[3].ints[0], attr[4].ints[0]
            layers.append([i.name, 'conv', [w[1], w[0], w[2], g, s, d, p]])
        elif i.op_type == 'Gemm':
            layers.append([i.name, 'dense', list(weights[-2].shape[::-1])])
        elif i.op_type == 'MaxPool':
            w, m, s = i.attribute[0].ints, i.attribute[1].ints, i.attribute[2].ints
            layers.append([i.name, 'maxpool', [list(i) for i in [w, m[2:], s]]])
        elif i.op_type == 'GlobalAveragePool':
            layers.append([i.name, 'gap', None])
        elif i.op_type == 'Upsample':
            mode, k = i.attribute[0].s.decode('utf-8'), weights.pop(-1)
            layers.append([i.name, 'upsample', [int(k[-1]), mode]])
        elif i.op_type == 'Flatten':
            layers.append([i.name, 'flatten', None])
        elif i.op_type == 'Unsqueeze':
            layers.append([i.name, 'unsqueeze', [i.attribute[0].ints[0]]])
        elif i.op_type == 'Relu':
            layers.append([i.name, 'relu', None])
        elif i.op_type == 'LeakyRelu':
            alpha = i.attribute[0].f
            layers.append([i.name, 'leakyrelu', [alpha]])
        elif i.op_type == 'Add':
            layers.append([i.name, 'add', None])
        elif i.op_type == 'Div':
            layers.append([i.name, 'div', None])
        elif i.op_type == 'Constant':
            buf = i.attribute[0].t.raw_data
            tp = i.attribute[0].t.data_type
            tp, fmt = [(int, 'int64'), (float, 'float32')][tp==1]
            const[i.output[0]] = tp(np.frombuffer(buf, fmt))
        elif i.op_type == 'Pow':
            flows[-1][0], k = flows[-1][0]
            layers.append([i.name, 'pow', [const[k]]])
        elif i.op_type == 'ReduceSum':
            axis, keep = i.attribute[0].ints[0], i.attribute[1].i
            layers.append([i.name, 'reducesum', [axis, keep]])
        elif i.op_type == 'Concat':
            layers.append([i.name, 'concat', [i.attribute[0].i]])
        elif i.op_type == 'Pad':
            layers.append([i.name, 'identity', None])
        elif i.op_type == 'Sigmoid':
            layers.append([i.name, 'sigmoid', None])
        elif i.op_type == 'AveragePool':
            for attr in i.attribute:
                if 'stride' in attr.name:
                    s = attr.ints[0]
                if 'kernel' in attr.name:
                    w = attr.ints[0]
            layers.append([i.name, 'avgpool', [w, s]])
        elif i.op_type == 'Shape':
            layers.append([i.name, 'shape', None])
        elif i.op_type == 'Gather':
            flows[-1][0], k = flows[-1][0]
            layers.append([i.name, 'gather', [const[k]]])
        elif i.op_type == 'Mul':
            layers.append([i.name, 'mul', None])
        elif i.op_type == 'Reshape':
            layers.append([i.name, 'reshape', None])
        elif i.op_type == 'Transpose':
            layers.append([i.name, 'transpose', [list(i.attribute[0].ints)]])
        elif i.op_type == 'LogSoftmax':
            layers.append([i.name, 'logsoftmax', [i.attribute[0].i]])
        else:
            print('lost layer:', i.name)
            return i
    layers.append(['return', 'return', None])
    flows.append([[i.name for i in graph.output], ['return'], 'plrst'])
    weights = np.hstack([i.ravel() for i in weights])

    np.save(path.replace('onnx', 'npy'), weights)
    with open(path.replace('onnx', 'json'), 'w') as f:
        json.dump({'layers':layers, 'flow':flows}, f)

    with zipfile.ZipFile(path.replace('onnx', 'pla'), 'w') as f:
        f.write(path.replace('onnx','json'))
        f.write(path.replace('onnx','npy'))
    
    

def torch2planer(net, name, x, in_name=None, out_name=None):
    import torch
    import sys
    np.save(name+'.npy', get_weight(net))
    stdout = sys.stdout
    sys.stdout = open(name+'.txt', 'w')
    torch.onnx.export(net, x, 'useless.onnx', verbose=True,
        input_names=in_name, output_names=out_name)
    sys.stdout = stdout
    body, flow = read_onnx(name)
    # the json file is needed
    with open(name+'.json', 'w') as jsfile:
        json.dump({'layers':body, 'flow':flow}, jsfile)

if __name__ == '__main__':
    a, b = read_onnx('../demo/yolov3-planer-2/yolov3')
