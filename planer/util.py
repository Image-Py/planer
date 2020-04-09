import cupy as np
import numpy 
from time import time

njit = None

def neighbors(shape, core, offset=0, dilation=1):
    shp = [slice(0,i) for i in core]
    idx = np.mgrid[tuple(shp)]
    idx = idx.reshape((len(core),-1))
    # print(core, type(offset), offset)
    offset = (np.array(core)-1)//2*np.array(offset)
    idx -= offset.reshape((-1,1))
    
    idx.T[:] *= np.array(dilation)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx.T, acc[::-1])

def jit_fill_col(pdimg, msk, idx, colimg):
    s = 0
    for i in range(len(pdimg)):
        if not msk[i]: continue
        for j in idx:
            colimg[s] = pdimg[i+j]
            s += 1
    return colimg

def fill_col(pdimg, msk, idx, colimg):
    rc = np.where(msk)[0]
    rc = rc.reshape((-1,1))+idx
    colimg[:] = pdimg[rc.ravel()]
    return colimg


def conv(img, core, group=1, stride=(1,1), dilation=(1,1), buf=['']):
    # print('conv in:', img.shape, img.mean())
    strh, strw = stride
    cimg_w = int(np.cumprod(core.shape[1:])[-1] * group)
    n,c,h,w = img.shape
    cimg_h = n*(h//strh)*(w//strw)
    if len(buf[0])<cimg_h*cimg_w:
        shp = cimg_h*cimg_w
        col_img = np.empty(int(shp), dtype=np.float32)
        
        buf[0] =  col_img
    else:
        col_img = buf[0][:cimg_h*cimg_w]
        col_img[:] = 0
    (n,c,h,w), (dh, dw) = core.shape, dilation
    shp = ((0,0),(0,0),(h*dh//2,h*dh//2),(w*dw//2,w*dw//2))
    pdimg = np.pad(img, shp, 'constant', constant_values=0)
    msk = np.zeros(pdimg.shape, dtype=np.bool)
    sliceh = slice(h*dh//2, -(h*dh//2) or None, strh)
    slicew = slice(w*dw//2, -(w*dw//2) or None, strw)
    msk[:, ::c, sliceh, slicew] = True
    
    nbs = neighbors(pdimg.shape[1:], core.shape[1:], (0,1,1), (1, dh, dw))
    a = time()
    fill_col(pdimg.ravel(), msk.ravel(), nbs, col_img)
    b = time()
    if group==1: 
        col_core = core.reshape((core.shape[0],-1))
        col_img = col_img.reshape((cimg_h, cimg_w))
        rst = col_core.dot(col_img.T)
    else:
        col_core = core.reshape((group, core.shape[0]//group,-1))
        col_img = col_img.reshape((group, -1, cimg_w//group))
        rst = [i.dot(j.T) for i,j in zip(col_core, col_img)]
        rst = np.concatenate(rst)
    c = time()
    # print('im2col:', b-a, 'dot', c-b)
    ni, ci, hi, wi = img.shape
    # print('conv out:', rst.mean())

    return rst.reshape((ni, n, hi//strh, wi//strw))

def jit_fill_max(pdimg, msk, idx, colimg):
    s = 0
    for i in range(len(pdimg)):
        if not msk[i]: continue
        for j in idx:
            colimg[s] = max(colimg[s], pdimg[i+j])
        s += 1
    return colimg

def jit_fill_mean(pdimg, msk, idx, colimg):
    s, l = 0, len(idx)
    for i in range(len(pdimg)):
        if not msk[i]: continue
        for j in idx:
            colimg[s] += pdimg[i+j]
        colimg[s] /= l
        s += 1
    return colimg

def fill_max(pdimg, msk, idx, colimg):
    rc = np.where(msk)[0]
    rc = rc.reshape((-1,1))+idx
    vs = pdimg[rc.ravel()].reshape((-1, len(idx)))
    np.max(vs, axis=-1, out=colimg)

def fill_mean(pdimg, msk, idx, colimg):
    rc = np.where(msk)[0]
    rc = rc.reshape((-1,1))+idx
    vs = pdimg[rc.ravel()].reshape((-1, len(idx)))
    np.mean(vs, axis=-1, out=colimg)


def pool(img, f, core=(2,2), stride=(2,2)):
    # print(img.shape, img.mean())
    start = time()
    (n,c,h,w), (ch, cw), (strh, strw) = img.shape, core, stride
    shp = ((0,0),(0,0),((ch-1)//2,)*2,((cw-1)//2,)*2)
    if np.array(shp).sum()==0: pdimg = img
    else: pdimg = np.pad(img, shp, 'constant', constant_values=0)
    
    msk = np.zeros(pdimg.shape, dtype=np.bool)
    sliceh = slice((ch-1)//2, -((ch-1)//2) or None, strh)
    slicew = slice((cw-1)//2, -((cw-1)//2) or None, strw)
    msk[:, :, sliceh, slicew] = True
    
    # nbs = neighbors(pdimg.shape[1:], (1,)+core, (0,1,1))
    nbs = neighbors(pdimg.shape, (1,1)+core, (0,0,1,1))
    colimg = np.zeros((n, c, h//strh, w//strw), dtype=np.float32)
    f(pdimg.ravel(), msk.ravel(), nbs, colimg.ravel())
    # print('pool time:', time()-start)
    return colimg

maxpool = lambda i, c=(2,2), s=(2,2): pool(i, fill_max, c, s)
avgpool = lambda i, c=(2,2), s=(2,2): pool(i, fill_mean, c, s)

def jit_bilinear(img, ra, rb, rs, _rs, ca, cb, cs, _cs, out):
    h, w = img.shape
    for r in range(out.shape[0]):
        rar = ra[r]
        rbr = rar+1
        rsr = rs[r]
        _rsr = _rs[r]
        for c in range(out.shape[1]):
            cac = ca[c]
            cbc = cac+1
            rra = img[rar,cac]*_rsr
            rra += img[rbr,cac]*rsr
            rrb = img[rar,cbc]*_rsr
            rrb += img[rbr,cbc]*rsr
            rcab = rra * _cs[c] + rrb * cs[c]
            out[r,c] = rcab

def bilinear(img, ra, rb, rs, _rs, ca, cb, cs, _cs, out):
    # print(ca, _cs, cb, cs)
    buf = img[:,ca]*_cs + img[:,cb]*cs
    out[:] = (buf[ra].T*_rs + buf[rb].T*rs).T

if not njit is None: bilinear = njit(jit_bilinear)

# def resize(img, size, out=None):
#     nc, (h, w) = img.shape[:-2], img.shape[-2:]
#     kh, kw = size[0]/h, size[1]/w
#     if out is None:
#         out = np.zeros(nc+size, dtype=img.dtype)
#     slicer = -0.5+0.5/kh, h-0.5-0.5/kh, size[0]
#     rs = np.linspace(*slicer, dtype=np.float32)
#     slicec = -0.5+0.5/kw, w-0.5-0.5/kw, size[1]
#     cs = np.linspace(*slicec, dtype=np.float32)
#     rs = np.clip(rs, 0, h-1)
#     cs = np.clip(cs, 0, w-1)
#     ra = np.floor(rs).astype(np.uint32)
#     ca = np.floor(cs).astype(np.uint32)
#     ra = np.clip(ra, 0, h-1.5)
#     ca = np.clip(ca, 0, w-1.5)
#     rs -= ra; cs -= ca; 
#     outcol = out.reshape((-1, *size))
#     imgcol = img.reshape((-1, h, w))
#     for i, o in zip(imgcol, outcol):
#         bilinear(i, ra, ra+1, rs, 1-rs, ca, ca+1, cs, 1-cs, o)
#     return out
def resize(img, size, out=None):
    nc, (h, w) = img.shape[:-2], img.shape[-2:]
    kh, kw = size[0]/h, size[1]/w
    if out is None:
        out = np.zeros(nc+size, dtype=img.dtype)
    slicer = -0.5+0.5/kh, h-0.5-0.5/kh, size[0]
    rs = np.linspace(*slicer, dtype=np.float32)
    slicec = -0.5+0.5/kw, w-0.5-0.5/kw, size[1]
    cs = np.linspace(*slicec, dtype=np.float32)
    np.clip(rs, 0, h-1, out=rs)
    np.clip(cs, 0, w-1, out=cs)
    ra = np.floor(np.clip(rs, 0, h-1.5)).astype(np.uint32)
    ca = np.floor(np.clip(cs, 0, w-1.5)).astype(np.uint32)
    rs -= ra; cs -= ca; 
    outcol = out.reshape((-1, *size))
    imgcol = img.reshape((-1, h, w))
    for i, o in zip(imgcol, outcol):
        bilinear(i, ra, ra+1, rs, 1-rs, ca, ca+1, cs, 1-cs, o)
    return out
    
def upsample(img, k, out=None):
    return resize(img, (img.shape[-2]*k, img.shape[-1]*k), out)

if __name__ == '__main__':
    from skimage.data import camera, astronaut
    import matplotlib.pyplot as plt
    from scipy.ndimage import convolve
    import cv2
    '''
    img = np.zeros((10,10,512,512), dtype=np.float32)
    # upsample(img, 2.0) # 0.4, 2.1
    img = astronaut().transpose((2,0,1))
    resize(img, (1024,1024))
    start = time()
    rst = resize(img, (300,300))
    print(time()-start)
    plt.imshow(rst.transpose((1,2,0)))
    plt.show()
    '''
               
    
    img = np.zeros((1, 128, 256, 256), dtype=np.float32)
    #img.ravel()[:] = np.arange(3*512*512)
    
    #core.ravel()[:] = np.arange(3*3*3*32)

    
    start = time()
    rst1 = resize(img, (512, 512))
    for i in range(10):
        rst1 = resize(img, (512, 512))
    print('cupy cost:', time()-start)

    import numpy as np

    img = np.zeros((256, 256, 128), dtype=np.float32)
    #img.ravel()[:] = np.arange(3*512*512)
    
    #core.ravel()[:] = np.arange(3*3*3*32)

    
    start = time()
    rst1 = cv2.resize(img, (512, 512))
    for i in range(10):
        rst1 = cv2.resize(img, (512, 512))
    
    print('numpy cost:', time()-start)
    
    
    ##nbs = neighbors((4, 5, 6), (3,3,3), offset = (0,1,1), dilation=(1,2,2))
    ##print(nbs)
    #arr = np.arange(16).reshape((1,1,4,4)).astype(np.float32)
    #rst = maxpool(arr, (2,2), (2,2))
    
    '''
    arr = np.zeros((3,4,5), dtype=np.bool)
    arr[0,:,:] = 1
    pd = np.pad(arr, ((0,0), (1,1), (1,1)), 'constant', constant_values=0)

    msk = np.zeros(pd.shape, dtype=np.bool)
    msk[0,1:-1,1:-1] = 1
    '''
