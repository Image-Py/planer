import sys
sys.path.append('../../')
import planer
from planer import read_net, resize
import cupy
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from time import time
from skimage import io

pal = planer.core(cupy)

img = io.imread('dog.jpg')[:, :, ::-1]
x = pal.array(img.astype('float32')/255.0)
x = x.transpose(2, 0, 1)[None, :, :, :]
x = resize(x, (416, 416))

anchors_all = pal.array([10, 13,  16, 30,  33, 23,  30, 61,  62,
                          45,  59, 119,  116, 90,  156, 198,  373, 326]).reshape(9, 2)

def sigmoid(x):
    return 1/(1+pal.exp(-x))

def post_process(fmps, anchors, thresh=0.4):

    # batch results
    bbox = {}
    for i in range(3):
        y = fmps[i]
        anchor = anchors[6-3*i:9-3*i, :]
        # gride size
        g = 13*2**i
        scale = 416/g
        grid_x = pal.tile(pal.arange(g), (g, 1)).reshape(1, 1, g, g).astype('float32')
        grid_y = pal.tile(pal.arange(g)[:, None], (1, g)).reshape(1, 1, g, g).astype('float32')

        n = y.shape[0]
        # 80 classes + 5
        y = y.reshape(n, 3, 85, g, g)

        obj = sigmoid(y[:, :, 4, :, :])
        cx = (sigmoid(y[:, :, 0, :, :]) + grid_x)*scale
        cy = (sigmoid(y[:, :, 1, :, :]) + grid_y)*scale
        w = pal.exp(y[:, :, 2, :, :]) * anchor[:, 0].reshape(-1, 3, 1, 1)
        h = pal.exp(y[:, :, 3, :, :]) * anchor[:, 1].reshape(-1, 3, 1, 1)
        
        pred = pal.argmax(sigmoid(y[:, :, 5:, :, :]), axis=2)

        for j in range(n):
            index_filt = obj>thresh
            obj_f = obj[index_filt][:, None]
            cx_f = cx[index_filt][:, None]
            cy_f = cy[index_filt][:, None]
            w_f = w[index_filt][:, None]
            h_f = h[index_filt][:, None]
            pred_f = pred[index_filt][:, None]
            x1 = cx_f - w_f//2
            y1 = cy_f - h_f//2
            x2 = cx_f + w_f//2
            y2 = cy_f + h_f//2
            box = pal.concatenate([obj_f, x1, y1, x2, y2, pred_f], axis=-1)
            if j in bbox.keys():
                bbox[j] = pal.concatenate([bbox[j], box], axis=0)
            else:
                bbox[j] = box
    return bbox

def nms(dets, thresh):
    x1 = dets[:, 1]
    y1 = dets[:, 2]
    x2 = dets[:, 3]
    y2 = dets[:, 4]
    scores = dets[:, 0]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(dets[i, :])
        xx1 = pal.maximum(x1[i], x1[order[1:]])
        yy1 = pal.maximum(y1[i], y1[order[1:]])
        xx2 = pal.minimum(x2[i], x2[order[1:]])
        yy2 = pal.minimum(y2[i], y2[order[1:]])

        w = pal.maximum(0.0, xx2 - xx1 + 1)
        h = pal.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = pal.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


net = read_net('yolov3')
y = net(x)
net.timer = {}

start = time()
y = net(x)
print('yolo-v3 time:', time()-start)

for k in net.timer:
    print(k, net.timer[k])

plt.imshow(img[:, :, ::-1])

# get one result from the batch results
bbox = post_process(y, anchors_all)[0]
start = time()
bbox = post_process(y, anchors_all)[0]
print('post process time:', time()-start)

keep = nms(bbox, 0.32)
start = time()
keep = nms(bbox, 0.32)
print('nms time:', time()-start)

for bbx in keep:
    currentAxis=plt.gca()
    x1 = pal.cpu(bbx[1])
    y1 = pal.cpu(bbx[2])
    x2 = pal.cpu(bbx[3])
    y2 = pal.cpu(bbx[4])
    rect=patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
    currentAxis.add_patch(rect)

plt.show()


