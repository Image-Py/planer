from torchvision.models import resnet18
import torch
import numpy as np
import torch.nn as nn
from time import time

net = resnet18(pretrained=True)
net.eval()
    
def export_npy(module, name='base'):
    weights = []  
    for m in module.modules():
        keys = {nn.Conv2d, nn.BatchNorm2d, nn.Linear}
        if m.__class__ == nn.Conv2d: 
            if m.bias is None:
                m.bias = nn.Parameter(torch.zeros(m.weight.shape[0]))
                # print(m.bias.shape)
        if not m.__class__ in keys : continue
        for p in m.parameters():
            weights.append(p.data.detach().cpu().numpy().ravel())
        if not isinstance(m, nn.BatchNorm2d): continue
        weights.append(m.running_mean.cpu().numpy().ravel())
        weights.append(m.running_var.cpu().numpy().ravel())
    np.save(name+'.npy', np.concatenate(weights))

# export weights into npy
export_npy(net, 'resnet18')

dummy_input = torch.randn(1, 3, 224, 224, device='cpu')
input_names = [ "input1" ]
output_names = [ "output1" ]

# export onnx info, the useless.onnx is useless, the verbose is what we needed.
torch.onnx.export(net, dummy_input, "useless.onnx", verbose=True, input_names=input_names, output_names=output_names)
