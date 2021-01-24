import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch 

class MLP(nn.Module):
    def __init__(self): 
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return F.sigmoid(x)

model = MLP()

weights = []
for param in model.parameters():
    print(param.data.shape)
    weights.append(param.data.numpy().ravel())
    
weights = np.concatenate(weights)
np.save('mlp.npy', weights)

x = torch.Tensor(4, 10).zero_()
print(model(x).data.numpy())