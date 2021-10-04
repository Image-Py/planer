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

def onnx2planer(path, zip=True):
    import numpy as np
    import onnx, onnx.numpy_helper
    graph = onnx.load(path).graph
    input_para = [i.name for i in graph.input]
    layers, inits, weights, flows, values = [], [], [], [], {}
    for i in graph.initializer: 
        v = onnx.numpy_helper.to_array(i)
        values[i.name] = len(weights), v.shape
        inits.append([i.name, v.shape, str(v.dtype)])
        weights.append(v)
    for i in graph.node:
        inpara = [j for j in i.input]
        outpara = [j for j in i.output]

        if len(inpara)==1: inpara = inpara[0]
        if len(outpara)==1: outpara = outpara[0]

        flows.append([inpara, [i.name], outpara])
        # weights.extend([values[i] for i in initpara])

        if i.op_type == 'BatchNormalization':
            cur = flows[-1]
            k, b, m, v = [weights[values[cur[0][j]][0]] for j in (1,2,3,4)]
            v_inv = 1/np.sqrt(v + 1e-5)
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
            attr, w = i.attribute, values[i.input[1]][1]
            g, d, p, s = attr[1].i, list(attr[0].ints), list(attr[3].ints)[-2:], list(attr[4].ints)
            layers.append([i.name, 'conv', {'shp':[w[1], w[0], w[2], w[3]], 'group':g, 'strides':s, 'dilation':d, 'pads':p}])
        elif i.op_type == 'Gemm':
            layers.append([i.name, 'dense', {'shp':list(values[i.input[1]][1][::-1])}])
        elif i.op_type == 'MaxPool':
            ks = ['kernel_shape', 'pads', 'strides']
            names = [j.name for j in i.attribute]
            w, m, s = [i.attribute[names.index(j)].ints for j in ks]
            layers.append([i.name, 'maxpool', {'w':list(w), 'pads':list(m[2:]), 'strides':list(s)}])
        elif i.op_type == 'GlobalAveragePool':
            layers.append([i.name, 'gap', {}])
        elif i.op_type == 'Upsample':
            idx = values[i.input[1]][0]
            inits[idx][2] = 'int64'
            weights[idx] = weights[idx].astype(np.int64)
            mode = i.attribute[0].s.decode('utf-8')
            layers.append([i.name, 'upsample', {'mode':mode}])
        elif i.op_type == 'Resize':
            flows[-1][0] = flows[-1][0][0]
            mode = i.attribute[2].s.decode()
            layers.append([i.name, 'upsample', {'mode':mode}])
        elif i.op_type == 'Flatten':
            layers.append([i.name, 'flatten', {}])
        elif i.op_type == 'Unsqueeze':
            layers.append([i.name, 'unsqueeze', {'dim':i.attribute[0].ints[0]}])
        elif i.op_type == 'Relu':
            layers.append([i.name, 'relu', {}])
        elif i.op_type == 'LeakyRelu':
            alpha = i.attribute[0].f
            layers.append([i.name, 'leakyrelu', {'alpha':alpha}])
        elif i.op_type == 'Add':
            layers.append([i.name, 'add', {}])
        elif i.op_type == 'Div':
            layers.append([i.name, 'div', {}])
        elif i.op_type == 'Constant':
            # print(i.name, i.output[0])
            dim = i.attribute[0].t.dims

            buf = i.attribute[0].t.raw_data
            tp = types[i.attribute[0].t.data_type]
            if len(buf)==0: continue
            v = np.frombuffer(buf, tp).reshape(dim).tolist()
            layers.append([i.name, 'const', {'value':v, 'dtype':tp}])
        elif i.op_type == 'Pow':
            layers.append([i.name, 'pow', {}])
        elif i.op_type == 'ReduceSum':
            axis, keep = i.attribute[0].ints[0], i.attribute[1].i
            layers.append([i.name, 'reducesum', {'axis':axis, 'keep_dim':keep}])
        elif i.op_type == 'Concat':
            layers.append([i.name, 'concat', {'axis':i.attribute[0].i}])
        elif i.op_type == 'Pad':
            layers.append([i.name, 'identity', {}])
        elif i.op_type == 'Sigmoid':
            layers.append([i.name, 'sigmoid', {}])
        elif i.op_type == 'AveragePool':
            print('AveragePool IO, need review')
            for attr in i.attribute:
                if 'stride' in attr.name: s = attr.ints[0]
                if 'kernel' in attr.name: w = attr.ints[0]
            layers.append([i.name, 'avgpool', [w, s]])
        elif i.op_type == 'Shape':
            layers.append([i.name, 'shape', {}])
        elif i.op_type == 'Gather':
            layers.append([i.name, 'gather', {'axis':i.attribute[0].i}])
        elif i.op_type == 'Mul':
            layers.append([i.name, 'mul', {}])
        elif i.op_type == 'Reshape':
            layers.append([i.name, 'reshape', {}])
        elif i.op_type == 'Transpose':
            layers.append([i.name, 'transpose', {'axis':list(i.attribute[0].ints)}])
        elif i.op_type == 'LogSoftmax':
            layers.append([i.name, 'logsoftmax', {'axis':i.attribute[0].i}])
        elif i.op_type == 'ConstantOfShape': 
            dim = i.attribute[0].t.dims
            buf = i.attribute[0].t.raw_data
            tp = types[i.attribute[0].t.data_type]
            v = np.frombuffer(buf, tp).tolist()[0]
            layers.append([i.name, 'constantofshape', {'value':v, 'dtype':tp}])
        elif i.op_type == 'Split': 
            layers.append([i.name, 'split', {'indices':list(i.attribute[1].ints), 'axis':i.attribute[0].i}])
        elif i.op_type == 'Tanh': 
            layers.append([i.name, 'tanh', {}])
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
        else:
            print('lost layer:', i.name)
            return i
    layers.append(['return', 'return', {}])
    flows.append([[i.name for i in graph.output], ['return'], 'plrst'])
    weights = np.hstack([i.view(dtype=np.uint8).ravel() for i in weights])

    np.save(path.replace('onnx', 'npy'), weights)
    with open(path.replace('onnx', 'json'), 'w') as f:
        json.dump({'input':input_para, 'inits':inits, 'layers':layers, 'flow':flows}, f)

    if zip:
        with zipfile.ZipFile(path.replace('onnx', 'pla'), 'w') as f:
            f.write(path.replace('onnx','json'))
            f.write(path.replace('onnx','npy'))
        os.remove(path.replace('onnx','json'))
        os.remove(path.replace('onnx','npy'))

if __name__ == '__main__':
    a, b = read_onnx('../demo/yolov3-planer-2/yolov3')
