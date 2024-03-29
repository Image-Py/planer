import json, re, os
import numpy, numpy as np
from .net import Net
from time import time
import json, zipfile
from io import BytesIO

def read_net(path, debug=False):
    net = Net()
    path = path.replace('.onnx', '')
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
    elif os.path.exists(path+'.onnx'):
        body, weights = read_onnx(path+'.onnx')
        if body == 'lost': return weights
        lay, flw = body['layers'], body['flow']
        inputs, inits = body['input'], body['inits']
    else: 
        return print('model %s not found!'%path)
    net.load_json(inputs, inits, lay, flw, debug)
    net.load_weights(weights)
    return net

types = [None, 'float32', 'uint8', 'int8', 'uint16', 'int16', 'int32', 'int64', 
    'str', 'bool', 'float16', 'float64', 'uint32', 'uint64', 'complex64', 'complex128']

def node(attrs, name, k=None, para=None): 
    node = None
    for i in attrs: 
        if i.name==name: node = i
    if k is None or node is None: 
        return node
    rst = getattr(node, k)
    if k=='ints': rst = list(rst)
    if k=='s': rst = rst.decode()
    if not para is None: 
        para[name] = rst
    return rst


def read_onnx(path):
    import onnx, onnx.numpy_helper
    graph = onnx.load(path).graph
    input_para = [i.name for i in graph.input]
    layers, inits, weights, flows, values = [], [], [], [], {}
    for i in graph.initializer: 
        v = onnx.numpy_helper.to_array(i)
        values[i.name] = len(weights), v.shape
        inits.append([i.name, v.shape, str(v.dtype)])
        if v.ndim==0: v = np.array([v])
        weights.append(v)
    for i in graph.node:
        inpara = [j for j in i.input]
        outpara = [j for j in i.output]



        if len(inpara)==1: inpara = inpara[0]
        if len(outpara)==1: outpara = outpara[0]

        flows.append([inpara, [i.name], outpara])
        # weights.extend([values[i] for i in initpara])
        # print(i.op_type, '===')
        if i.op_type == 'BatchNormalization':
            cur = flows[-1]
            k, b, m, v = [weights[values[cur[0][j]][0]] for j in (1,2,3,4)]
            v_inv = 1/numpy.sqrt(v + 1e-5)
            kmv_inv_b = -k*m*v_inv + b
            kv_inv = k*v_inv
            kmv_inv_b.shape = kv_inv.shape = (1,-1,1,1)
            
            kname, bname = cur[0][1] + '_invK', cur[0][1] + '_invB'
            values[kname] = len(weights), kv_inv.shape
            values[bname] = len(weights)+1, kmv_inv_b.shape
            inits.append([kname, kv_inv.shape, str(kv_inv.dtype)])
            inits.append([bname, kmv_inv_b.shape, str(kmv_inv_b.dtype)])
            weights.extend([kv_inv, kmv_inv_b])
            cur[0] = [cur[0][0], kname, bname]
            layers.append([i.name, 'batchnorm', {}])
        elif i.op_type == 'Conv':
            # attr, w = i.attribute, values[i.input[1]][1]
            attr = i.attribute
            g = node(attr, 'group', 'i') or 1
            d = node(attr, 'dilations', 'ints')
            p = node(attr, 'pads', 'ints')
            s = node(attr, 'strides', 'ints')
            layers.append([i.name, 'conv', {
                'group':g, 'strides':s, 'dilations':d, 'pads':p}])
        elif i.op_type == 'ConvTranspose':
            attr = i.attribute
            para = {}
            g = node(attr, 'group', 'i', para)
            d = node(attr, 'dilations', 'ints', para)
            p = node(attr, 'pads', 'ints', para)
            s = node(attr, 'strides', 'ints', para)
            op = node(attr, 'output_padding', 'ints', para)
            layers.append([i.name, 'convtranspose', para])
        elif i.op_type == 'Gemm':
            layers.append([i.name, 'dense', {'shp':list(values[i.input[1]][1][::-1])}])
        elif i.op_type == 'MaxPool':
            w = node(i.attribute, 'kernel_shape', 'ints')
            m = node(i.attribute, 'pads', 'ints')
            s = node(i.attribute, 'strides', 'ints')
            layers.append([i.name, 'maxpool', {'w':w, 'pads':m, 'strides':s}])
        elif i.op_type == 'GlobalAveragePool':
            layers.append([i.name, 'gap', {}])
        elif i.op_type == 'Upsample':
            mode = node(i.attribute, 'mode', 's')
            layers.append([i.name, 'upsample', {'mode':mode}])
        elif i.op_type == 'Resize':
            mode = node(i.attribute, 'mode', 's')
            nearest_mode = node(i.attribute, 'nearest_mode', 's')
            trans_mode = node(i.attribute, 'coordinate_transformation_mode', 's')
            layers.append([i.name, 'resize', {'mode':mode, 'nearest_mode':nearest_mode, 
                'coordinate_transformation_mode': trans_mode}])
        elif i.op_type == 'Flatten':
            layers.append([i.name, 'flatten', {}])
        elif i.op_type == 'Unsqueeze':
            axis = node(i.attribute, 'axes', 'ints')
            layers.append([i.name, 'unsqueeze', {} if axis is None else {'axes':axis}])
        elif i.op_type == 'Squeeze':
            axis = node(i.attribute, 'axes', 'ints')
            layers.append([i.name, 'squeeze', {} if axis is None else {'axes':axis}])
        elif i.op_type == 'Relu':
            layers.append([i.name, 'relu', {}])
        elif i.op_type == 'LeakyRelu':
            alpha = i.attribute[0].f
            layers.append([i.name, 'leakyrelu', {'alpha':alpha}])
        elif i.op_type == 'HardSigmoid':
            para = {}
            node(i.attribute, 'alpha', 'f', para)
            node(i.attribute, 'beta', 'f', para)
            layers.append([i.name, 'hardsigmoid', para])
        elif i.op_type == 'Add':
            layers.append([i.name, 'add', {}])
        elif i.op_type == 'Sub':
            layers.append([i.name, 'sub', {}])
        elif i.op_type == 'Div':
            layers.append([i.name, 'div', {}])
        elif i.op_type == 'Tile':
            layers.append([i.name, 'tile', {}])
        elif i.op_type == 'MatMul':
            layers.append([i.name, 'matmul', {}])
        elif i.op_type == 'Constant':
            _, _, name = flows.pop(-1)
            dim = i.attribute[0].t.dims
            tp = types[i.attribute[0].t.data_type]

            v = onnx.numpy_helper.to_array(i.attribute[0].t)
            values[name] = len(weights), v.shape
            inits.append([name, v.shape, str(v.dtype)])
            if v.ndim==0: v = np.array([v])
            weights.append(v)
            #layers.append([i.name, 'const', {'value':v, 'dtype':tp}])
        elif i.op_type == 'Identity':
            layers.append([i.name, 'identity', {}])
        elif i.op_type == 'Pow':
            layers.append([i.name, 'pow', {}])
        elif i.op_type == 'ReduceSum':
            para = {}
            node(i.attribute, 'axes', 'ints', para)
            node(i.attribute, 'keepdims', 'i', para)
            layers.append([i.name, 'reducesum', para])
        elif i.op_type == 'ReduceMean':
            para = {}
            node(i.attribute, 'axes', 'ints', para)
            node(i.attribute, 'keepdims', 'i', para)
            layers.append([i.name, 'reducemean', para])
        elif i.op_type == 'ReduceMax':
            para = {}
            node(i.attribute, 'axes', 'ints', para)
            node(i.attribute, 'keepdims', 'i', para)
            layers.append([i.name, 'reducemax', para])
        elif i.op_type == 'ReduceMin':
            para = {}
            node(i.attribute, 'axes', 'ints', para)
            node(i.attribute, 'keepdims', 'i', para)
            layers.append([i.name, 'reducemin', para])
        elif i.op_type == 'Concat':
            layers.append([i.name, 'concat', {'axis':i.attribute[0].i}])
        elif i.op_type == 'Pad':
            para = {}
            node(i.attribute, 'mode', 's', para)
            node(i.attribute, 'constant_value', 'f', para)
            layers.append([i.name, 'pad', para])
        elif i.op_type == 'Sigmoid':
            layers.append([i.name, 'sigmoid', {}])
        elif i.op_type == 'AveragePool':
            w = node(i.attribute, 'kernel_shape', 'ints')
            m = node(i.attribute, 'pads', 'ints')
            s = node(i.attribute, 'strides', 'ints')
            layers.append([i.name, 'averagepool', {'w':w, 'pads':m, 'strides':s}])
        elif i.op_type == 'LSTM':
            para = {'hidden_size': i.attribute[0].i}
            node(i.attribute, 'direction', 's', para)
            layers.append([i.name, 'lstm', para])
        elif i.op_type == 'Shape':
            layers.append([i.name, 'shape', {}])
        elif i.op_type == 'Gather':
            layers.append([i.name, 'gather', {'axis':node(i.attribute, 'axis', 'i') or 0}])
        elif i.op_type == 'Mul':
            layers.append([i.name, 'mul', {}])
        elif i.op_type == 'Reshape':
            layers.append([i.name, 'reshape', {}])
        elif i.op_type == 'Transpose':
            layers.append([i.name, 'transpose', {'axis':node(i.attribute, 'perm', 'ints')}])
        elif i.op_type == 'LogSoftmax':
            layers.append([i.name, 'logsoftmax', {'axis':i.attribute[0].i}])
        elif i.op_type == 'Softmax':
            layers.append([i.name, 'softmax', {'axis':i.attribute[0].i}])
        elif i.op_type == 'ConstantOfShape': 
            v = onnx.numpy_helper.to_array(i.attribute[0].t)
            tp, v = str(v.dtype), v.tolist()
            v = v[0] if len(v)==1 else 0
            layers.append([i.name, 'constantofshape', {'value':v, 'dtype':tp}])
        elif i.op_type == 'Greater':
            layers.append([i.name, 'greater', {}])
        elif i.op_type == 'NonZero':
            layers.append([i.name, 'nonzero', {}])
        elif i.op_type == 'GreaterOrEqual':
            layers.append([i.name, 'greaterorequal', {}])
        elif i.op_type == 'TopK':
            para = {}
            node(i.attribute, 'axis', 'i', para)
            node(i.attribute, 'largest', 'i', para)
            node(i.attribute, 'sorted', 'i', para)
            layers.append([i.name, 'topk', para])
        elif i.op_type == 'Split': 
            split = node(i.attribute, 'split', 'ints')
            para = {'axis': node(i.attribute, 'axis', 'i')}
            if not split is None: para['split'] = split
            layers.append([i.name, 'split', para])
        elif i.op_type == 'Tanh': 
            layers.append([i.name, 'tanh', {}])
        elif i.op_type == 'Exp': 
            layers.append([i.name, 'exp', {}])
        elif i.op_type == 'Log': 
            layers.append([i.name, 'log', {}])
        elif i.op_type == 'Slice':
            layers.append([i.name, 'slice', {}])
        elif i.op_type == 'Expand':
            layers.append([i.name, 'expand', {}])
        elif i.op_type == 'Equal':
            layers.append([i.name, 'equal', {}])
        elif i.op_type == 'Cast':
            layers.append([i.name, 'cast', {'dtype':types[i.attribute[0].i]}])
        elif i.op_type == 'Range':
            layers.append([i.name, 'range', {}])
        elif i.op_type == 'Where':
            layers.append([i.name, 'where', {}])
        elif i.op_type == 'ScatterND':
            layers.append([i.name, 'scatternd', {}])
        elif i.op_type == 'InstanceNormalization':
            layers.append([i.name, 'instancenormalization', {'epsilon':i.attribute[0].f}])
        elif i.op_type == 'Sqrt':
            layers.append([i.name, 'sqrt', {}])
        elif i.op_type == 'Erf':
            layers.append([i.name, 'erf', {}])
        elif i.op_type=='Reciprocal':
            layers.append([i.name, 'erf', {}])
        elif i.op_type == 'Clip':
            minv = node(i.attribute, 'min', 'f')
            maxv = node(i.attribute, 'max', 'f')
            para = {}
            if minv: para['min']=minv
            if maxv: para['max']=maxv
            layers.append([i.name, 'clip', para])
        else:
            print('lost layer:', i.op_type)
            return 'lost', i

    layers.append(['return', 'return', {}])
    flows.append([[i.name for i in graph.output], ['return'], 'plrst'])
    weights = np.hstack([i.view(dtype=np.uint8).ravel() for i in weights])
    return {'input':input_para, 'inits':inits, 'layers':layers, 'flow':flows}, weights

def onnx2pla(path, zip=True):
    graph, weights = read_onnx(path)
    np.save(path.replace('onnx', 'npy'), weights)
    with open(path.replace('onnx', 'json'), 'w') as f:
        json.dump(graph, f)
    if zip:
        with zipfile.ZipFile(path.replace('onnx', 'pla'), 'w') as f:
            f.write(path[:-4]+'json', os.path.split(path)[1][:-4]+'json')
            f.write(path[:-4]+'npy', os.path.split(path)[1][:-4]+'npy')
        os.remove(path.replace('onnx','json'))
        os.remove(path.replace('onnx','npy'))

if __name__ == '__main__':
    a, b = read_onnx('../demo/yolov3-planer-2/yolov3')
