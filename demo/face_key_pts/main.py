import sys
sys.path.append('../../')
import planer
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from time import time
import numpy as np

# get pixels on the line
def line(r1, c1, r2, c2):
    d = max(abs(r1-r2), abs(c1-c2)) + 1
    rs = np.linspace(r1,r2,d).round()
    cs = np.linspace(c1,c2,d).round()
    return rs.astype(int), cs.astype(int)

# count flow from kay points
# weights: ear - face - jaw 's weights, 0 means stay there.
# fac: + means fat, - means thin, 0 would got a zeros flow
def count_flow(rc, weights=[0,1,2,2,2,2,1,1,0], fac=5):
    flow = np.zeros((224,224,2), dtype=np.float32)
    l = rc[:17].astype(np.int)
    nose = rc[31:36].mean(axis=0)
    dv = nose - l
    dv.T[:] *= weights + weights[1:]
    for s,e,v1,v2 in zip(l[:-1], l[1:], dv[:-1], dv[1:]):
        flow[line(*s, *e)] = (v1+v2)/2
    flow[:,:,0] = planer.gaussian_filter(flow[:,:,0], 20)
    flow[:,:,1] = planer.gaussian_filter(flow[:,:,1], 20)
    nv = np.linalg.norm(flow, axis=-1)
    return flow * (fac/nv.max())

# apply the flow
def flow_map(img, flow):
    des = np.mgrid[0:224, 0:224] + flow.transpose(2,0,1)
    des = planer.resize(des, img.shape[:2])
    des *= np.array(img.shape[:2]).reshape(2,1,1)/224
    rst = planer.mapcoord(img.transpose(2,0,1), *des*1)
    return rst.transpose(1,2,0).astype(np.uint8)

# generate check board
def check_board(shp, d=20):
    checkboard = np.zeros(shp, dtype=np.uint8)
    msk = np.tile([False]*d+[True]*d, 1000)
    checkboard[msk[:shp[0]]] += 128
    checkboard[:,msk[:shp[1]]] += 128
    return ((checkboard>0)*255).astype(np.uint8)

if __name__ == '__main__':
    net = planer.read_net('./face_key')
    face = imread('./face.jpg')
    x = np.array(face.transpose(2,0,1)[None,:,:,:])
    x = planer.resize(x, (224,224)).astype(np.float32)

    start = time()
    y = net(x/255)
    print(time()-start)

    rc = y.reshape(-1,2)[:,::-1] * 50 + 100

    thin = flow_map(face, count_flow(rc, fac=-10))
    fat = flow_map(face, count_flow(rc, fac=10))

    grid = check_board(face.shape, 30)
    thin_grid = flow_map(grid, count_flow(rc, fac=-10))
    fat_grid = flow_map(grid, count_flow(rc, fac=10))

    # 绘图
    plt.subplot(231).imshow(face)
    plt.subplot(232).imshow(thin)
    plt.subplot(233).imshow(fat)
    plt.subplot(234).imshow(grid)
    plt.subplot(235).imshow(thin_grid)
    plt.subplot(236).imshow(fat_grid)
    plt.show()
    
