from planer import *  

class CustomNet(Net):  
    def __init__(self):  
        self.fc1 = Dense(10, 20)  
        self.sigmoid = Sigmoid()  
        self.fc2 = Dense(20, 5)  
  
    def forward(self, x):  
        y = self.fc1(x)  
        y = self.sigmoid(y)  
        y = self.fc2(y)  
        return self.sigmoid(y)

model = CustomNet()

# 检查MLP维度
x = np.zeros((4, 10), dtype='float32')
y = model(x)
print(x.shape, y.shape)

# 检测是否与PyTorch结果相符
weights = np.load('mlp.npy')
# 游标检测，用来判断weights是否全部加载完
s1 = model.fc1.load(weights)
s2 = model.fc2.load(weights[s1:])
x = np.zeros((4, 10), dtype='float32')
y = model(x)
print(y)
