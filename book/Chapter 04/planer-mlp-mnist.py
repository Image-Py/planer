from planer import *  
import cv2

class CustomNet(Net):  
    def __init__(self):  
        self.fc1 = Dense(784, 100)  
        self.sigmoid = Sigmoid()  
        self.fc2 = Dense(100, 10)  
        self.softmax = Softmax()
        
    def forward(self, x):  
        y = self.fc1(x)  
        y = self.sigmoid(y)  
        y = self.fc2(y)  
        return self.softmax(y)

model = CustomNet()

# 检测是否与PyTorch结果相符
weights = np.load('mnist_mlp.npy')
# 游标检测，用来判断weights是否全部加载完
s1 = model.fc1.load(weights)
s2 = model.fc2.load(weights[s1:])

x = cv2.imread('2.png', 0).ravel()/255.0
print(x.shape)
y = model(x[None, :])
score = np.max(y)
pred = np.argmax(y)
print('class:', pred, 'score:', score)


