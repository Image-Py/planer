import cupy as np
from time import time


def neighbors(shape, core, offset=0, dilation=1):
    shp = [slice(0, i) for i in core]
    idx = np.mgrid[tuple(shp)]
    idx = idx.reshape((len(core), -1))
    offset = (np.array(core)-1)//2*np.array(offset)
    idx -= offset.reshape((-1, 1))
    idx.T[:] *= np.array(dilation)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx.T, acc[::-1])


def conv(img, core, group=1, stride=(1, 1), dilation=(1, 1), buf=['']):
    (strh, strw), (dh, dw) = stride, dilation
    (n, c, h, w), (ni, ci, hi, wi)  = core.shape, img.shape
    cimg_w = c * h * w * group
    cimg_h, imgs = ni*(hi//strh)*(wi//strw), []
    shp = ((0, 0), (0, 0), (dh*(h//2),)*2, (dw*(w//2),)*2)
    img = np.pad(img, shp, 'constant', constant_values=0)
    for r in range(0, h*dh, dh):
        for c in range(0, w*dw, dw):
            imgs.append(img[:,:,0+r:hi+r:strh, 0+c:wi+c:strw])
    imgs = [i[:,:,:,:,None] for i in imgs]
    col_img = np.concatenate(imgs, axis=-1)
    col_img.shape = (ni, group, ci, cimg_h, -1, w*h)
    col_img = col_img.transpose((0,1,3,4,2,5))
    if group == 1:
        col_core = core.reshape((core.shape[0], -1))
        col_img = col_img.reshape((cimg_h, cimg_w))
        rst = col_core.dot(col_img.T)
    else:
        col_core = core.reshape((group, core.shape[0]//group, -1))
        col_img = col_img.reshape((group, -1, cimg_w//group))
        rst = [i.dot(j.T) for i, j in zip(col_core, col_img)]
        rst = np.concatenate(rst)
    return rst.reshape((ni, n, hi//strh, wi//strw))


def pool_nxn(img, f, s):
    n, c, h, w = img.shape
    rshp = img.reshape(n,c,h//s,s,w//s,s)
    rshp = rshp.transpose((0,1,2,4,3,5))
    if f == 'max':
        return rshp.max(axis=(4,5))
    if f == 'mean':
        return rshp.mean(axis=(4,5))


def pool(img, f, core=(2, 2), stride=(2, 2)):
    (n, c, h, w), (ch, cw), (strh, strw) = img.shape, core, stride
    shp = ((0, 0), (0, 0), ((ch-1)//2,)*2, ((cw-1)//2,)*2)
    img = np.pad(img, shp, 'constant', constant_values=0)
    (imn, ic, ih, iw), imgs = img.shape, []
    for r in range(ch):
        for c in range(cw):
            imgs.append(img[:,:,r:h+r:strh,c:w+c:strw])
    col_img = np.concatenate([i[:,:,:,:,None] for i in imgs], axis=-1)
    if f == 'max': return col_img.max(axis=-1)
    if f == 'mean': return col_img.mean(axis=-1)


def pool(img, f, core=(2, 2), stride=(2, 2)):
    (n, c, h, w), (ch, cw), (strh, strw) = img.shape, core, stride
    shp = ((0, 0), (0, 0), ((ch-1)//2,)*2, ((cw-1)//2,)*2)
    img = np.pad(img, shp, 'constant', constant_values=0)

    imn, ic, ih, iw = img.shape
    acc = np.cumprod((1,)+img.shape[::-1][:-1])
    nbs = neighbors(img.shape[2:], core, (1, 1))
    slicew = np.arange((ch-1)//2, iw-((ch-1)//2), strw) * acc[0]
    sliceh = np.arange((cw-1)//2, ih-((cw-1)//2), strh) * acc[1]

    idx = sliceh.reshape((-1, 1)) + slicew.ravel()
    idx = idx.reshape((-1, 1)) + nbs
    colimg = img.reshape((imn*ic, -1))[:, idx.ravel()]
    colimg = colimg.reshape((-1, len(nbs)))
    if f == 'max': colimg = colimg.max(axis=1)
    if f == 'mean': colimg = colimg.mean(axis=1)
    return colimg.reshape((n, c, h//strh, w//strw))


def maxpool(i, c=(2, 2), s=(2, 2)): 
    if not c[0] == c[1] == s[0] == s[1]:
        return pool(i, 'max', c, s)
    return pool_nxn(i, 'max', s[0])


def avgpool(i, c=(2, 2), s=(2, 2)): 
    if not c[0] == c[1] == s[0] == s[1]:
        return pool(i, 'mean', c, s)
    return pool_nxn(i, 'mean', s[0])


def resize(img, size):
    nc, (h, w) = img.shape[:-2], img.shape[-2:]
    kh, kw = size[0]/h, size[1]/w
    slicer = -0.5+0.5/kh, h-0.5-0.5/kh, size[0]
    rs = np.linspace(*slicer, dtype=np.float32)
    slicec = -0.5+0.5/kw, w-0.5-0.5/kw, size[1]
    cs = np.linspace(*slicec, dtype=np.float32)

    np.clip(rs, 0, h-1, out=rs)
    np.clip(cs, 0, w-1, out=cs)
    r = np.floor(np.clip(rs, 0, h-1.5))
    c = np.floor(np.clip(cs, 0, w-1.5))
    r = r.astype(np.uint32)
    c = c.astype(np.uint32)

    rs -= r; cs -= c; _cs = 1-cs; _rs = 1-rs
    r.shape = rs.shape = _rs.shape = (-1, 1)
    img = img.reshape((-1, h*w))

    idx = (r * w + c).ravel()
    klt = (_rs * _cs).ravel()
    rst = img[:, idx] * klt

    idx += 1
    krt = (_rs * cs).ravel()
    rst += img[:, idx] * krt

    idx += w
    krb = (rs * cs).ravel()
    rst += img[:, idx] * krb

    idx -= 1
    klb = (rs * _cs).ravel()
    rst += img[:, idx] * klb

    return rst.reshape(nc + size)

def upsample(img, k, out=None):
    return resize(img, (img.shape[-2]*k, img.shape[-1]*k))


def make_upmat(k):
    xs = np.linspace(0.5/k, 1-0.5/k, k*1)
    rs, cs = xs[:,None], xs[None,:]
    klt = ((1-cs)*(1-rs)).reshape((1,-1))
    krt = (cs * (1-rs)).reshape((1,-1))
    klb = ((1-cs) * rs).reshape((1,-1))
    krb = (cs * rs).reshape((1,-1))
    return np.vstack([klt, krt, klb, krb])
    

def upsample(img, k, matbuf={}):    
    n, c, h, w = img.shape
    img = (img[:,:,:1,:], img, img[:,:,-1:,:])
    img = np.concatenate(img, axis=2)
    img = (img[:,:,:,:1], img, img[:,:,:,-1:])
    img = np.concatenate(img, axis=3)
    if not k in matbuf: matbuf[k] = make_upmat(k)    
    imgs = [img[:,:,:-1,:-1], img[:,:,:-1,1:],
            img[:,:,1:,:-1], img[:,:,1:,1:]]
    imgs = [i[:,:,:,:,None] for i in imgs]
    rst = np.concatenate(imgs, axis=-1)
    rst = np.dot(rst.reshape((-1,4)), matbuf[k])
    rst = rst.reshape((-1, w+1, k, k))
    rst = rst.transpose((0,2,1,3))
    rst = rst.reshape((n,c,(h+1)*k, (w+1)*k))
    return rst[:,:,k//2:-k//2,k//2:-k//2]

    
if __name__ == '__main__':
    img = np.zeros((1,1,128,128), dtype=np.float32)

    a = upsample(img, 4)
    b = upsample2(img, 4)
    
    start = time()
    for i in range(10):
        upsample(img, 4)
    print(time()-start)

    start = time()
    for i in range(10):
        upsample2(img, 4)
    print(time()-start)
    
    
