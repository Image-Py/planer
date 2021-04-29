import numpy as np
from time import time

def pad(img, shp, mode='constant', constant_values=0):
    if shp[2][0]==shp[2][1]==shp[3][0]==shp[3][1]==0: return img
    if mode != 'constant': return np.pad(img, shp, mode)
    (n, c, h, w), (mn, mc, mh, mw) = img.shape, shp
    newimg = np.zeros((n, c, h+mh[0]*2, w+mw[0]*2), dtype=img.dtype)
    newimg[:,:,mh[0]:h+mh[0],mw[0]:w+mw[0]] = img
    return newimg

def conv(img, core, group=1, mar=(1, 1), stride=(1, 1), dilation=(1, 1), mode='constant'):
    (strh, strw), (dh, dw) = stride, dilation
    (n, c, h, w), (ni, ci, hi, wi)  = core.shape, img.shape
    cimg_w = c * h * w * group
    cimg_h, i = (hi//strh)*(wi//strw), 0
    shp = ((0, 0), (0, 0), (mar[0],)*2, (mar[1],)*2)
    img = pad(img, shp, mode, constant_values=0)
    img = img.transpose((1,0,2,3)) # nchw -> cnhw
    nh = (hi + sum(shp[2]) - h + strh)//strh
    nw = (wi + sum(shp[3]) - w + strw)//strw
    nsh, nsw = nh * strh, nw * strw
    col_img = np.zeros((ci, w*h,  ni, nh, nw), img.dtype) #(h*w, c, N, H, W)
    for r in range(0, h*dh, dh):
        for c in range(0, w*dw, dw):
            col_img[:,i], i = img[:,:,0+r:nsh+r:strh, 0+c:nsw+c:strw], i+1
    col_core = core.reshape((group, core.shape[0]//group, -1))
    col_img.shape = (group, cimg_w//group, -1)
    rst = [i.dot(j) for i, j in zip(col_core, col_img)]
    rst = rst[0] if group==1 else np.concatenate(rst)
    return rst.reshape((n, ni, nh, nw)).transpose(1, 0, 2, 3)

def pool(img, f, core=(2, 2), mar=None, stride=(2, 2),  const=0):
    (n, c, h, w), (ch, cw), (strh, strw) = img.shape, core, stride
    shp = ((0, 0), (0, 0), (mar[0],)*2, (mar[1],)*2)
    img = pad(img, shp, 'constant', constant_values=0)
    (imn, ic, ih, iw), imgs = img.shape, []
    dh = (h + sum(shp[2]) - core[0] + strh)//strh
    dw = (w + sum(shp[3]) - core[1] + strw)//strw
    nsh, nsw = dh * strh, dw * strw
    buf = np.zeros(img.shape[:2]+(dh, dw), np.float32)
    if const != 0: buf[:] = const
    for r in range(0, ch, 1):
        for c in range(0, cw, 1):
            f(img[:,:,r:nsh+r:strh,c:nsw+c:strw], buf, out=buf)
    return buf

def maxpool(i, c=(2, 2), mar=(0,0), s=(2, 2)):
    return pool(i, np.maximum, c, mar, s, -1e4)

def avgpool(i, c=(2, 2), mar=(0,0), s=(2, 2)):
    rst = pool(i, np.add, c, mar, s, 0)
    rst /= c[0] * c[1]
    return rst
    
def resize(img, size):
    nc, (h, w) = img.shape[:-2], img.shape[-2:]
    kh, kw = size[0]/h, size[1]/w
    slicer = -0.5+0.5/kh, h-0.5-0.5/kh, size[0]
    rs = np.linspace(*slicer, dtype=np.float32)
    slicec = -0.5+0.5/kw, w-0.5-0.5/kw, size[1]
    cs = np.linspace(*slicec, dtype=np.float32)
    np.clip(rs, 0, h-1, out=rs)
    np.clip(cs, 0, w-1, out=cs)
    ra = np.floor(np.clip(rs, 0, h-1.5))
    ca = np.floor(np.clip(cs, 0, w-1.5))
    ra, ca = ra.astype(int), ca.astype(int)
    rs -= ra; cs -= ca; rb = ra+1; cb = ca+1;
    rs.shape, img.shape = (-1,1), (-1, h, w)
    buf = img[:,:,ca]*(1-cs) + img[:,:,cb]*cs
    result = buf[:,ra,:]*(1-rs) + buf[:,rb,:]*rs
    return result.reshape(nc + size)

def mapcoord(img, rs, cs):
    nc, (h, w) = img.shape[:-2], img.shape[-2:]
    np.clip(rs, 0, h-1, out=rs)
    np.clip(cs, 0, w-1, out=cs)
    ra = np.floor(np.clip(rs, 0, h-1.5))
    ca = np.floor(np.clip(cs, 0, w-1.5))
    ra, ca = ra.astype(int), ca.astype(int)
    rs -= ra; cs -= ca; rb = ra+1; cb = ca+1;
    img.shape = (-1, h, w)
    buf = img[:,ra,ca]*((1-cs) * (1-rs))
    buf += img[:,rb,cb] * (cs * rs)
    buf += img[:,ra,cb] * ((1-rs) * cs)
    buf += img[:,rb,ca] * ((1-cs) * rs)
    return buf

def make_upmat(k):
    xs = np.linspace(0.5/k, 1-0.5/k, k*1, dtype=np.float32)
    rs, cs = xs[:,None], xs[None,:]
    klt = ((1-cs)*(1-rs)).reshape((1,-1))
    krt = (cs * (1-rs)).reshape((1,-1))
    klb = ((1-cs) * rs).reshape((1,-1))
    krb = (cs * rs).reshape((1,-1))
    return np.vstack([klt, krt, klb, krb])
    
def upsample_blinear(img, k, matbuf={}):    
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

def upsample_nearest(img, k):
    n, c, h, w = img.shape
    rst = np.zeros((n, c, h, k, w, k), dtype=np.float32)
    trst = rst.transpose((0,1,2,4,3,5))
    trst[:] = img[:,:,:,:,None,None]
    return rst.reshape((n, c, h*k, w*k))

def upsample(img, k, mode):
    if mode=='nearest': return upsample_nearest(img, k)
    if mode=='linear': return upsample_blinear(img, k)

def conv_auto(img, core, mode='reflect'):
    shp, dim, (h, w) = img.shape, img.ndim, core.shape
    img = np.pad(img, ((h//2,h//2),(w//2,w//2),(0,0))[:dim], mode=mode)
    rst, buf = np.zeros((2,) + shp, dtype=np.float32)
    for r,c in np.mgrid[:h,:w].reshape(2,-1).T:
        buf[:] = img[r:r+shp[0],c:c+shp[1]]
        buf *= core[r,c]; rst += buf
    return rst       

def conv_rc(img, core_r, core_c, mode='reflect'):
    return conv_auto(conv_auto(img, core_r), core_c).astype(img.dtype)
    
def uniform_filter(img, size=3, mode='reflect'):
    core = np.ones(size, dtype=np.float32)/size
    return conv_rc(img, core[None,:], core[:,None], mode)
    
def gaussian_filter(img, sig=2, mode='reflect'):
    x = np.arange(-int(sig*2.5+0.5), int(sig*2.5+0.5)+1)
    core = np.exp(-x**2/2/sig**2)/sig/(2*np.pi)**0.5
    return conv_rc(img, core[None,:], core[:,None], mode)

if __name__ == '__main__':
    '''
    img = np.zeros((1, 64, 512, 512), dtype=np.float32)
    core = np.zeros((32, 64, 3, 3), dtype=np.float32)
    conv(img, core)
    '''
    import scipy.ndimage as ndimg
    import matplotlib.pyplot as plt
    from skimage.data import camera
    from time import time
    
    img = camera()
    img[2,2] = 10
    start = time()
    rst = gaussian_filter(img, 5)
    print(time()-start)
