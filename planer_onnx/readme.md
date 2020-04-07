### Converting model from PyTorch to Planer

* PyTorch 1.1.0 needed


We first convert the parameters of the pytorch module into numpy format by:
```python
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
```

Then we use the onnx converting verbose to get the graph and IR. We provide a example to convert torchvision resnet18 in ```trans_resnet18.py```.

Running ```python trans_resnet18.py > resnet18.txt``` will have two files ```resnet18.npy```, ```resnet18.txt```.

And we can have a simple code to verify the converted model by
```python
from planer import read_onnx
import numpy as np
# get the planer array library
pal = planer.core(np)
# the same folder should contain resnet18.txt, resnet18.npy
net = read_onnx('resnet18')
# input should be float32
x = pal.random.randn(1, 3, 224, 224).astype('float32')
y = net(x)
print(y.shape)
```
