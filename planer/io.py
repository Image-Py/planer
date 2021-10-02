import json, re, os
import numpy as np
from .net import Net
from time import time
import json, zipfile
from io import BytesIO

def read_net(path, debug=False):
    net = Net()
    if os.path.exists(path+'.pla'):
        with zipfile.ZipFile(path+'.pla') as f:
            path = os.path.split(path)[1]
            body = json.loads(f.read(path+'.json'))
            lay, flw = body['layers'], body['flow']
            inputs, inits = body['input'], body['inits']
            buf = BytesIO(f.read(path+'.npy'))
            weights = np.load(buf)
    elif os.path.exists(path+'.json'):
        with open(path+'.json') as f:
            body = json.load(f)
            lay, flw = body['layers'], body['flow']
            inputs, inits = body['input'], body['inits']
        weights = np.load(path+'.npy')
    else: return print('model %s not found!'%path)
    net.load_json(inputs, inits, lay, flw, debug)
    net.load_weights(weights)
    return net

types = [None, 'float32', 'uint8', 'int8', 'uint16', 'int16', 'int32', 'int64', 
    'str', 'bool', 'float16', 'float64', 'uint32', 'uint64', 'complex64', 'complex128']

def onnx2planer(path):
    import onnx, onnx.numpy_helper
    graph = onnx.load(path).graph
    input_para = [i.name for i in graph.input]
    layers, inits, weights, flows, values = [], [], [], [], {}
    for i in graph.initializer: 
        v = onnx.numpy_helper.to_array(i)
        values[i.name] = v
        inits.append([i.name, v.shape, str(v.dtype)])
        weights.append(v.view(dtype=np.uint8))
    for i in graph.node:
        inpara = [j for j in i.input]
        # subpara = [j for j in i.input if not j in values]
        outpara = [j for j in i.output]

        initset = {'BatchNormalization', 'Conv', 'Gemm', 'Resize', 'Upsample'}

        # if not i.op_type in initset: subpara = [j for j in i.input]
        # print(i.input, i.name, i.output, has, no)

        if len(inpara)==1: inpara = inpara[0]
        if len(outpara)==1: outpara = outpara[0]

        # no para layer has init input: constarray
        '''
        if len(initpara)>0 and not i.op_type in initset:
            for j in initpara:
                layers.append(['ConstArray_%s'%j, 'constarray', [values[j].shape]])
                flows.append([['None'], 'ConstArray_%s'%j, j])
        '''

        flows.append([inpara, [i.name], outpara])
        # weights.extend([values[i] for i in initpara])

        if i.op_type == 'BatchNormalization':
            layers.append([i.name, 'batchnorm', None])
        elif i.op_type == 'Conv':
            attr, w = i.attribute, values[i.input[1]].shape
            g, d, p, s = attr[1].i, list(attr[0].ints), list(attr[3].ints)[-2:], list(attr[4].ints)
            layers.append([i.name, 'conv', [w[1], w[0], [w[2], w[3]], g, s, d, p]])
        elif i.op_type == 'Gemm':
            layers.append([i.name, 'dense', list(values[i.input[1]].shape[::-1])])
        elif i.op_type == 'MaxPool':
            ks = ['kernel_shape', 'pads', 'strides']
            names = [j.name for j in i.attribute]
            w, m, s = [i.attribute[names.index(j)].ints for j in ks]
            layers.append([i.name, 'maxpool', [list(i) for i in [w, m[2:], s]]])
        elif i.op_type == 'GlobalAveragePool':
            layers.append([i.name, 'gap', None])
        elif i.op_type == 'Upsample':
            mode = i.attribute[0].s.decode('utf-8')
            layers.append([i.name, 'upsample', [mode]])
        elif i.op_type == 'Resize':
            return i
            flows[-1][0] = flows[-1][0][0]
            mode = i.attribute[2].s.decode()
            layers.append([i.name, 'upsample', [mode]])
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
            # print(i.name, i.output[0])
            dim = i.attribute[0].t.dims

            buf = i.attribute[0].t.raw_data
            tp = types[i.attribute[0].t.data_type]
            if len(buf)==0: continue
            v = np.frombuffer(buf, tp).reshape(dim).tolist()
            layers.append([i.name, 'const', [v, tp]])
        elif i.op_type == 'Pow':
            layers.append([i.name, 'pow', None])
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
                if 'stride' in attr.name: s = attr.ints[0]
                if 'kernel' in attr.name: w = attr.ints[0]
            layers.append([i.name, 'avgpool', [w, s]])
        elif i.op_type == 'Shape':
            layers.append([i.name, 'shape', None])
        elif i.op_type == 'Gather':
            layers.append([i.name, 'gather', [i.attribute[0].i]])
        elif i.op_type == 'Mul':
            layers.append([i.name, 'mul', None])
        elif i.op_type == 'Reshape':
            layers.append([i.name, 'reshape', None])
        elif i.op_type == 'Transpose':
            layers.append([i.name, 'transpose', [list(i.attribute[0].ints)]])
        elif i.op_type == 'LogSoftmax':
            layers.append([i.name, 'logsoftmax', [i.attribute[0].i]])
        elif i.op_type == 'ConstantOfShape': 
            dim = i.attribute[0].t.dims
            buf = i.attribute[0].t.raw_data
            tp = types[i.attribute[0].t.data_type]
            value = np.frombuffer(buf, tp).tolist()[0]
            layers.append([i.name, 'constantofshape', [value, tp]])
        elif i.op_type == 'Split': 
            layers.append([i.name, 'split', [list(i.attribute[1].ints), i.attribute[0].i]])
        elif i.op_type == 'Tanh': 
            layers.append([i.name, 'tanh', None])
        elif i.op_type == 'Slice':
            layers.append([i.name, 'slice', None])
        elif i.op_type == 'Expand':
            layers.append([i.name, 'expand', None])
        elif i.op_type == 'Equal':
            layers.append([i.name, 'equal', None])
        elif i.op_type == 'Cast':
            layers.append([i.name, 'cast', [types[i.attribute[0].i]]])
        elif i.op_type == 'Range':
            layers.append([i.name, 'range', None])
        elif i.op_type == 'Where':
            layers.append([i.name, 'where', None])
        elif i.op_type == 'ScatterND':
            layers.append([i.name, 'scatternd', None])
        else:
            print('lost layer:', i.name)
            return i
    layers.append(['return', 'return', None])
    flows.append([[i.name for i in graph.output], ['return'], 'plrst'])
    weights = np.hstack([i.ravel() for i in weights])

    np.save(path.replace('onnx', 'npy'), weights)
    with open(path.replace('onnx', 'json'), 'w') as f:
        json.dump({'input':input_para, 'inits':inits, 'layers':layers, 'flow':flows}, f)

    with zipfile.ZipFile(path.replace('onnx', 'pla'), 'w') as f:
        f.write(path.replace('onnx','json'))
        f.write(path.replace('onnx','npy'))

if __name__ == '__main__':
    a, b = read_onnx('../demo/yolov3-planer-2/yolov3')
