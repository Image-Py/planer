import json, re, os
import numpy as np
from .net import Net
from time import time
import json

def read_net(path):
    net = Net()
    with open(path+'.json') as f:
        body = json.load(f)
        lay, flw = body['layers'], body['flow']
    net.load_json(lay, flw)
    net.load_weights(np.load(path+'.npy'))
    return net

def parse(matched):
    gps = list(matched.groups())
    if len(matched.groups()) == 0:
        return ''
    if gps[0]=='return':
        gps.insert(0, 'plrst')
    for i in range(len(gps)):
        if '%' in gps[i]:
            gps[i] = gps[i].replace('%', "'")
            gps[i] = gps[i].replace(',', "',")
            gps[i] = gps[i].replace(')', "',)")
        elif gps[i][-1] == ')' and len(gps[i]) > 2:
            gps[i] = gps[i][:-1]+',)'

    return str(gps)+'\n'

conv = re.compile(
    r'.*%(.+?) .+?(Conv).+?dilations=(\[\d+?, \d+?\]).+?group=(\d+).+?strides=(\[\d+?, \d+?\]).+?(\(%.+?, %.+?, %.+?\)).+?\n')
relu = re.compile(r'.*%(.+?) .+?(Relu)\(%(.+?)\).+?\n')
leakyrelu = re.compile(
    r'.*%(.+?) .+?(LeakyRelu)\[alpha=(.+?)\]\(%(.+?)\).+?\n')
gap = re.compile(r'.*%(.+?) .+?(GlobalAveragePool)\(%(.+?)\).+?\n')
sigmoid = re.compile(r'.*%(.+?) .+?(Sigmoid)\(%(.+?)\).+?\n')
maxpool = re.compile(
    r'.*%(.+?) .+?(MaxPool).+?kernel_shape=(\[\d+?, \d+?\]).+?strides=(\[\d+?, \d+?\]).+?\(%(.+?)\).+?\n')
avgpool = re.compile(
    r'.*%.+?Pad.+?\n.+?%(.+?) .+?(AveragePool).+?kernel_shape=(\[\d+?, \d+?\]).+?strides=(\[\d+?, \d+?\]).+?\(%(.+?)\).+?\n')
upsample = re.compile(
    r'.*%(.+?) .+?(Upsample)\[mode="(.+?)"\](\(%.+?, %.+?\)).+?\n')   
unsqueeze = re.compile(r'.*%(.+?) .+?(Unsqueeze)\[axes=(\[.+?\])\]\(%(.+?)\).+?\n')
flatten = re.compile(r'.*%(.+?) .+?(Flatten)\[axis=(\d+?)\]\(%(.+?)\).+?\n')
dense = re.compile(r'.*%(.+?) .+?(Gemm).+(\(%.+?, %.+?, %.+?\)).+?\n')
concat = re.compile(r'.*%(.+?) .+?(Concat).+(\(%.+?\)).+?\n')
batchnorm = re.compile(r'.*%(.+?) .+?(BatchNormalization).+?(\(.+?\)).+?\n')
add = re.compile(r'.*%(.+?) .+?(Add)(\(%.+?\)).+?\n')
pow = re.compile(r'.*%(.+?) .+?(Pow)(\(%.+?\)).+?\n')
div = re.compile(r'.*%(.+?) .+?(Div)(\(%.+?\)).+?\n')
reducesum = re.compile(r'.*%(.+?) .+?(ReduceSum)\[axes=(\[.+?\]), keepdims=(\d+?)\]\(%(.+?)\).+?\n')
mul = re.compile(r'.*%(.+?) .+?(Mul)(\(%.+?\))\n')
const = re.compile(r'.*%(.+?) .+?(Constant).*value=\{(.+?)\}.+?\n')
weight = re.compile(r'.*%(.+?) .+?(\(.*?\)).*\n')
output = re.compile(r'.*(return) (\(%.+?\))')

res = (flatten, upsample, conv, relu, leakyrelu, gap, sigmoid, maxpool,
       avgpool, dense, unsqueeze, reducesum, concat, pow, add, mul, div, const, batchnorm, weight, output)

def read_onnx(path):
    with open(path+'.txt') as f: cont = f.read()
    cont = re.compile(r'\d+?:\d+').sub(lambda x:x.group(0).split(':')[0], cont)
    cont = re.compile(r'(, )?requires_grad=\d|(, )?device=cpu').sub('', cont)
    for i in res: cont = i.sub(parse, cont)
    # for i in cont.split('\n'): print(i)
    cont = [eval(i) for i in cont.split('\n') if len(i) > 0 and i[0] in '[']
    cont = [[eval(j) if (',' in j) else j for j in i] for i in cont]        

    body, flow, key, same = [], [], {},{}
    for i in cont:
        num = len(body)
        if len(i) == 2:
            key[i[0]] = i[1]
        elif i[1] == 'Conv':
            name = i[5][1]+'-'+i[5][2]
            if name in same:
                flow.append((i[5][0], [same[name]], i[0]))
                continue
            same[name] = 'conv_%s' % num
            shp = [key[i[5][1]][j]
                for j in (1, 0, 2)] + [int(i[3]), i[4][0], i[2][0]]
            body.append(('conv_%s' % num, 'conv', shp))
            flow.append((i[5][0], ['conv_%s' % num], i[0]))
        elif i[1] == 'Gemm':
            body.append(('dense_%s' % num, 'dense', key[i[2][1]][::-1]))
            flow.append((i[2][0], ['dense_%s' % num], i[0]))
        elif i[1] == 'Sigmoid':
            body.append(('sigmoid_%s' % num, 'sigmoid', None))
            flow.append((i[2], ['sigmoid_%s' % num], i[0]))
        elif i[1] == 'Relu':
            body.append(('relu_%s' % num, 'relu', None))
            flow.append((i[2], ['relu_%s' % num], i[0]))
        elif i[1] == 'LeakyRelu':
            body.append(('leakyrelu_%s' % num, 'leakyrelu', [float(i[2])]))
            flow.append((i[3], ['leakyrelu_%s' % num], i[0]))
        elif i[1] == 'GlobalAveragePool':
            body.append(('gap_%s' % num, 'gap', None))
            flow.append((i[2], ['gap_%s' % num], i[0]))
        elif i[1] == 'Add':
            body.append(('add_%s' % num, 'add', None))
            flow.append((i[2], ['add_%s' % num], i[0]))
        elif i[1] == 'Pow':
            body.append(('pow_%s' % num, 'pow', None))
            flow.append((i[2], ['pow_%s' % num], i[0]))
        elif i[1] == 'Mul':
            body.append(('mul_%s' % num, 'mul', None))
            flow.append((i[2], ['mul_%s' % num], i[0]))
        elif i[1] == 'Div':
            body.append(('div_%s' % num, 'div', None))
            flow.append((i[2], ['div_%s' % num], i[0]))
        elif i[1] == 'Constant':
            body.append(('const_%s' % num, 'const', [float(i[2])]))
            flow.append(('None', ['const_%s' % num], i[0]))
        elif i[1] == 'Concat':
            body.append(('concat_%s' % num, 'concat', None))
            flow.append((i[2], ['concat_%s' % num], i[0]))
        elif i[1] == 'AveragePool':
            body.append(('avgpool_%s' % num, 'avgpool', [i[2][0], i[3][0]]))
            flow.append((str(int(i[4])-1), ['avgpool_%s' % num], i[0]))
        elif i[1] == 'MaxPool':
            body.append(('maxpool_%s' % num, 'maxpool', [i[2][0], i[3][0]]))
            flow.append((i[4], ['maxpool_%s' % num], i[0]))
        elif i[1] == 'Upsample':
            body.append(('upsample_%s' % num, 'upsample', [2, i[2]]))
            flow.append((i[3][0], ['upsample_%s' % num], i[0]))
        elif i[1] == 'BatchNormalization':
            body.append(('batchnorm_%s' % num, 'batchnorm', [key[i[2][1]][0]]))
            flow.append((i[2][0], ['batchnorm_%s' % num, ], i[0]))
        elif i[1] == 'Flatten':
            body.append(('flatten_%s' % num, 'flatten', None))
            flow.append((i[3], ['flatten_%s' % num], i[0]))
        elif i[1] == 'Unsqueeze':
            body.append(('unsqueeze_%s' % num, 'unsqueeze', [eval(i[2])]))
            flow.append((i[3], ['unsqueeze_%s' % num], i[0]))
        elif i[1] == 'ReduceSum':
            body.append(('reducesum_%s' % num, 'reducesum', [eval(i[2]), int(i[3])]))
            flow.append((i[4], ['reducesum_%s' % num], i[0]))
        elif i[1] == 'return':
            body.append(('return_%s' % num, 'return', None))
            out = i[2] if len(i[2])>1 else i[2][0]
            flow.append((out, ['return_%s' % num], i[0]))
    return body, flow

def get_weight(module):
    import torch
    from torch import nn
    weights = []  
    for m in module.modules():
        keys = {nn.Conv2d, nn.BatchNorm2d, nn.Linear}
        if m.__class__ == nn.Conv2d: 
            if m.bias is None:
                m.bias = nn.Parameter(torch.zeros(m.weight.shape[0]))
        if not m.__class__ in keys : continue
        for p in m.parameters():
            weights.append(p.data.detach().cpu().numpy().ravel())
        if not isinstance(m, nn.BatchNorm2d): continue
        weights.append(m.running_mean.cpu().numpy().ravel())
        weights.append(m.running_var.cpu().numpy().ravel())
    return np.concatenate(weights)

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
