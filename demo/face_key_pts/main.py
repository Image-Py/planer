import sys
sys.path.append('../../')
import planer
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from time import time
import numpy as np
import scipy.ndimage as ndimg
from skimage.draw import line

net = planer.read_net('./face_key')

x = np.zeros((1,3,224,224), dtype=np.float32)
face = imread('./face.jpg')


x = np.array(face.transpose(2,0,1)[None,:,:,:])
x = planer.resize(x, (224,224)).astype(np.float32)

start = time()
y = net(x/255)
print(time()-start)

rc = y.reshape(-1,2)[:,::-1] * 50 + 100
# rc = rc * face.shape[:2][::-1] / 224

# 以鼻子为中心，脸轮廓向鼻子收敛
flow = np.zeros((224,224,2), dtype=np.float32)
l = rc[:17].astype(np.int)
nose = rc[31:36].mean(axis=0)
dv = nose - l
# 规定收敛强度，耳朵，下巴最低点是0，中间渐变
dv.T[:] *= [0,1,2,2,2,2,1,1,0,1,1,2,2,2,2,1,0]
for s,e,v1,v2 in zip(l[:-1], l[1:], dv[:-1], dv[1:]):
    flow[line(*s, *e)] = (v1+v2)/2
# 高斯模糊，生成连续场
flow[:,:,0] = ndimg.gaussian_filter(flow[:,:,0], 20)
flow[:,:,1] = ndimg.gaussian_filter(flow[:,:,1], 20)
# 调整强度控制，0为不变，负数收缩，整数扩张
nv = np.linalg.norm(flow, axis=-1)
flow *= 10/nv.max()

# 生成标准网格，减去位移场
des = np.mgrid[0:224, 0:224]
des = des + flow.transpose(2,0,1)
# 复原到原图尺寸
flow = planer.resize(des, face.shape[:2])
flow *= np.array(face.shape[:2]).reshape(2,1,1)/224
# 插值得到结果
mapc = ndimg.map_coordinates
rst = [mapc(face[:,:,i], flow, mode='nearest') for i in (0,1,2)]
rst = np.concatenate([i[:,:,None] for i in rst], -1)

# 绘图
plt.subplot(121).imshow(face)
plt.subplot(122).imshow(rst)
plt.show()
plt.show()

