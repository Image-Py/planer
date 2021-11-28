import numpy, numpy as np
from time import time

def pad(img, shp, mode='constant', constant_values=0):
    if shp[2][0]==shp[2][1]==shp[3][0]==shp[3][1]==0: return img
    if mode != 'constant': return np.pad(img, shp, mode)
    (n, c, h, w), (mn, mc, mh, mw) = img.shape, shp
    newimg = np.zeros((n, c, h+mh[0]*2, w+mw[0]*2), dtype=img.dtype)
    newimg[:,:,mh[0]:h+mh[0],mw[0]:w+mw[0]] = img
    return newimg

from concurrent.futures import ThreadPoolExecutor

conv_buf = []
def clear_buf(): global conv_buf; conv_buf = []

def conv_for(img, core, group=1, pads=(1, 1, 1, 1), strides=(1, 1), dilation=(1, 1), mode='constant'):
    threadPool = ThreadPoolExecutor(max_workers=9) # for 3x3 core
    (strh, strw), (dh, dw) = strides, dilation
    (n, c, h, w), (ni, ci, hi, wi)  = core.shape, img.shape
    cimg_w = c * h * w * group
    cimg_h, i = (hi//strh)*(wi//strw), 0
    shp = (0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])
    img = pad(img, shp, mode, constant_values=0)
    nh = (hi + sum(shp[2]) - (h-1)*dh-1 + strh)//strh
    nw = (wi + sum(shp[3]) - (w-1)*dw-1 + strw)//strw
    nsh, nsw = nh * strh, nw * strw
    # ================ img 2 col ================
    global conv_buf
    img = img.transpose((1,0,2,3)) # nchw -> cnhw
    size = ci * w * h * ni * nh * nw
    if len(conv_buf) < size: conv_buf = np.zeros(size, dtype=np.float32) 
    col_img = conv_buf[:size].reshape(ci, w*h,  ni, nh, nw) #(h*w, c, N, H, W)
    def set_value(img, i, v): img[:,i] = v
    for r in range(0, h*dh, dh):
        for c in range(0, w*dw, dw):
            im, i = img[:,:,0+r:nsh+r:strh, 0+c:nsw+c:strw], i+1
            threadPool.submit(set_value, col_img, i-1, im)
    threadPool.shutdown(wait=True)
    # ============================================
    col_core = core.reshape((group, core.shape[0]//group, -1))
    col_img = col_img.reshape(group, cimg_w//group, -1)
    rst = [i.dot(j) for i, j in zip(col_core, col_img)]
    rst = rst[0] if group==1 else np.concatenate(rst)
    return rst.reshape((n, ni, nh, nw)).transpose(1, 0, 2, 3)

def conv_stride(img, core, group=1, pads=(1, 1, 1, 1), strides=(1, 1), dilation=(1, 1), mode='constant'):
    (strh, strw), (dh, dw) = strides, dilation
    (n, c, h, w), (ni, ci, hi, wi)  = core.shape, img.shape
    cimg_w = c * h * w * group
    cimg_h, i = (hi//strh)*(wi//strw), 0
    shp = (0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])
    img = pad(img, shp, mode, constant_values=0)
    nh = (hi + sum(shp[2]) - (h-1)*dh-1 + strh)//strh
    nw = (wi + sum(shp[3]) - (w-1)*dw-1 + strw)//strw
    nsh, nsw = nh * strh, nw * strw
    # ================ img 2 col ================
    ss, shape = img.strides, (ci, w, h,  ni, nh, nw)
    strides  = (ss[-3], ss[-2]*dh, ss[-1]*dw, ss[-4], ss[-2]*strh, ss[-1]*strw)
    col_img = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    # ============================================
    col_core = core.reshape(group, core.shape[0]//group, -1)
    col_img = col_img.reshape(group, cimg_w//group, -1)
    rst = [i.dot(j) for i, j in zip(col_core, col_img)]
    rst = rst[0] if group==1 else np.concatenate(rst)
    return rst.reshape((n, ni, nh, nw)).transpose(1, 0, 2, 3)

def conv_dnn(img, core, bias=None, group=1, pads=(1, 1, 1, 1), strides=(1, 1), dilation=(1, 1), mode='constant'):
    (strh, strw), (dh, dw) = strides, dilation
    (n, c, h, w), (ni, ci, hi, wi)  = core.shape, img.shape
    shp = (0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])
    if pads[0]==pads[2] and pads[1]==pads[3]: pads = (pads[0], pads[2])
    else: img, pads = pad(img, shp, mode, constant_values=0), (0,0)
    nh = (hi + sum(shp[2]) - (h-1)*dh-1 + strh)//strh
    nw = (wi + sum(shp[3]) - (w-1)*dw-1 + strw)//strw    
    y = np.zeros((ni, n, nh, nw), dtype='float32')
    dnn.convolution_forward(img, core, bias, y, pads, 
        (strides[0], strides[1]), (dilation[0], dilation[1]), 1, auto_tune=True, tensor_core='always')
    return y

def pool(img, f, core=(2, 2), pads=(0,0,0,0), stride=(2, 2),  const=0):
    (n, c, h, w), (ch, cw), (strh, strw) = img.shape, core, stride
    shp = ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3]))
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

def maxpool(i, c=(2, 2), mar=(0,0,0,0), s=(2, 2)):
    return pool(i, np.maximum, c, mar, s, -1e4)

def avgpool(i, c=(2, 2), mar=(0,0,0,0), s=(2, 2)):
    rst = pool(i, np.add, c, mar, s, 0)
    rst /= c[0] * c[1]
    return rst

def make_upmat(k):
    ys = np.linspace(0.5/k[0], 1-0.5/k[0], k[0]*1, dtype=np.float32)
    xs = np.linspace(0.5/k[1], 1-0.5/k[1], k[1]*1, dtype=np.float32)
    rs, cs = ys[:,None], xs[None,:]
    if k[0]==1: return np.vstack([1-xs, xs])
    if k[1]==1: return np.vstack([1-ys, ys])
    klt = ((1-cs)*(1-rs)).reshape((1,-1))
    krt = (cs * (1-rs)).reshape((1,-1))
    klb = ((1-cs) * rs).reshape((1,-1))
    krb = (cs * rs).reshape((1,-1))
    return np.vstack([klt, krt, klb, krb])
    
def upsample_blinear(img, k):
    n, c, h, w = img.shape
    if k[0] == k[1] == 1: return img
    if k[0]>1:
        img = (img[:,:,:1,:], img, img[:,:,-1:,:])
        img = np.concatenate(img, axis=2)
    if k[1]>1:
        img = (img[:,:,:,:1], img, img[:,:,:,-1:])
        img = np.concatenate(img, axis=3)
    imgs = [img[:,:,:-1,:-1], img[:,:,:-1,1:],
            img[:,:,1:,:-1], img[:,:,1:,1:]]
    if k[0]==1: imgs = [img[:,:,:,:-1], img[:,:,:,1:]]
    if k[1]==1: imgs = [img[:,:,:-1,:], img[:,:,1:,:]]
    imgs = [i[:,:,:,:,None] for i in imgs]
    rst = np.concatenate(imgs, axis=-1)
    rst = np.dot(rst.reshape((-1,len(imgs))), make_upmat(k))
    hh, ww = h + (k[0]>1), w + (k[1]>1)
    rst = rst.reshape((-1, ww, k[0], k[1]))
    rst = rst.transpose((0,2,1,3))
    rst = rst.reshape((n, c ,hh*k[0], ww*k[1]))
    return rst[:,:,k[0]//2:h*k[0]+k[0]//2,k[1]//2:w*k[1]+k[1]//2]

def offset(k, trans_mode, round_mode):
    idx = np.arange(-64, 64)
    if trans_mode == 'half_pixel':
        idx = (idx + 0.5)/k - 0.5
    if trans_mode == 'asymmetric':
        idx = idx / k
    if round_mode == 'round_prefer_floor':
        idx = np.round(idx-1e-3)
    if round_mode == 'round_prefer_ceil':
        idx = np.round(idx+1e-3)
    if round_mode == 'ceil':
        idx = np.ceil(idx)
    if round_mode == 'floor':
        idx = np.floor(idx)
    idx = idx.astype(np.int16)
    return np.argmax(idx==0)-64

def pix_offset(img, dr, dc):
    n, c, h, w = img.shape
    if dr == dc == 0: return img
    if dr>=0: sr1s, sr1e, sr2s, sr2e, sr, rr = dr, h, 0, h-dr, (0, dr), 0
    else: sr1s, sr1e, sr2s, sr2e, sr, rr = 0, h+dr, -dr, h, (h+dr, h), h-1
    if dc>=0: sc1s, sc1e, sc2s, sc2e, sc, cc = dc, w, 0, w-dc, (0, dc), 0
    else: sr1r, sr1e, sr2s, sr2e, sc, cc = 0, w+dc, -dc, w, (w+dc, w), w-1
    img[:, :, sr1s:sr1e, sc1s:sc1e] = img[:, :, sr2s:sr2e, sc2s:sc2e]
    img[:, :, slice(*sr), :] = img[:, :, rr:rr+1, :]
    img[:, :, :, slice(*sc)] = img[:, :, :, cc:cc+1]
    return img

def upsample_nearest(img, k, trans_mode='half-pixel', round_mode='round_prefer_ceil'):
    n, c, h, w = img.shape
    rst = np.zeros((n, c, h*k[0], w*k[1]), dtype=img.dtype)
    for r in range(k[0]):
        for c in range(k[1]):
            rst[:,:,r::k[0],c::k[1]]=img

    offr, offc = [offset(i, trans_mode, round_mode) for i in k]
    return pix_offset(rst, offr, offc)

def upsample(img, k, mode, trans_mode='half-pixcel', round_mode='round_prefer_ceil'):
    if mode=='nearest': return upsample_nearest(img, k, trans_mode, round_mode)
    if mode=='linear': return upsample_blinear(img, k)
    
# ===== below is some image process function =====
import math, itertools

def conv_auto(img, core, mode='reflect', keeptp=True):
    shp, dim, (h, w) = img.shape, img.ndim, core.shape
    img = np.pad(img, ((h//2,h//2),(w//2,w//2),(0,0))[:dim], mode=mode)
    rst, buf = np.zeros((2,) + shp, dtype=np.float32)
    for r,c in np.mgrid[:h,:w].reshape(2,-1).T:
        buf[:] = img[r:r+shp[0],c:c+shp[1]]
        buf *= core[r,c]; rst += buf
    return rst.astype(img.dtype) if keeptp else rst

def conv_rc(img, core_r, core_c, mode='reflect'):
    return conv_auto(conv_auto(img, core_r, keeptp=False), core_c)
    
def uniform_filter(img, size=3, mode='reflect'):
    core = np.ones(size, dtype=np.float32)/size
    return conv_rc(img, core[None,:], core[:,None], mode)
    
def gaussian_filter(img, sig=2, mode='reflect'):
    x = np.arange(-int(sig*2.5+0.5), int(sig*2.5+0.5)+1)
    core = np.exp(-x**2/2/sig**2)/sig/(2*np.pi)**0.5
    return conv_rc(img, core[None,:], core[:,None], mode)

def make_slice(l, w, mar):
    r = np.linspace(0, l-w, math.ceil((l-mar)/(w-mar)))
    return [slice(i, i+w) for i in r.astype(int).tolist()]

def grid_slice(H, W, h, w, mar):
    a, b = make_slice(H, h, mar), make_slice(W, w, mar)
    return list(itertools.product(a, b))

def resize(img, size, backend=None):
    nn = backend or np
    d, (h, w) = img.ndim, img.shape[:2]
    kh, kw = size[0]/h, size[1]/w
    slicer = -0.5+0.5/kh, h-0.5-0.5/kh, size[0]
    rs = nn.linspace(*slicer, dtype=nn.float32)
    slicec = -0.5+0.5/kw, w-0.5-0.5/kw, size[1]
    cs = nn.linspace(*slicec, dtype=nn.float32)
    nn.clip(rs, 0, h-1, out=rs)
    nn.clip(cs, 0, w-1, out=cs)
    ra = nn.floor(nn.clip(rs, 0, h-1.5))
    ca = nn.floor(nn.clip(cs, 0, w-1.5))
    ra, ca = ra.astype(int), ca.astype(int)
    rs -= ra; cs -= ca; rb = ra+1; cb = ca+1;
    rs.shape, cs.shape = (-1,1,1)[:d], (1,-1,1)[:d]
    buf = img[:,ca]*(1-cs) + img[:,cb]*cs
    return buf[ra,:]*(1-rs) + buf[rb,:]*rs

def mapcoord(img, rs, cs, keeptp=True, backend=None):
    nn = backend or np
    d, (h, w) = img.ndim, img.shape[:2]
    nn.clip(rs, 0, h-1, out=rs)
    nn.clip(cs, 0, w-1, out=cs)
    ra = nn.floor(nn.clip(rs, 0, h-1.5))
    ca = nn.floor(nn.clip(cs, 0, w-1.5))
    ra, ca = ra.astype(int), ca.astype(int)
    rs -= ra; cs -= ca; rb = ra+1; cb = ca+1;
    if d==3: rs, cs = rs[:,:,None], cs[:,:,None]
    buf = img[ra,ca]*((1-cs) * (1-rs))
    buf += img[rb,cb] * (cs * rs)
    buf += img[ra,cb] * ((1-rs) * cs)
    buf += img[rb,ca] * ((1-cs) * rs)
    return buf.astype(img.dtype) if keeptp else buf

# sample：float or tuple, float means factor tuple means size
# glob：force adjust image to glob's integral multiple.
# window：after sample, if larger than window, then tiled by window
# margin：overlay between window, float means factor and int means width
def tile(sample=1, glob=1, window=1024, margin=0.1, astype='float32', progress=print):
    def wrapf(f):
        def wrap(*p, **key):
            (h, w), ori_img = p[0].shape[:2], p[0]
            samecore = isinstance(ori_img, np.ndarray)
            img = np.asarray(ori_img, dtype=astype)
            tps = {'sample', 'window', 'glob', 'margin', 'progress'}
            ftp = fp, tp = {}, {}
            for i in key: ftp[i in tps][i] = key[i]
            ssz = tp.get('sample', sample)
            wsz = wsh = wsw = tp.get('window', window)
            gsz = tp.get('glob', glob)
            mar = tp.get('margin', margin)
            info = tp.get('progress', progress)
            if isinstance(ssz, tuple): ssz = list(ssz)
            else: ssz = [int(h*ssz), int(w*ssz)]
            # smaller than window, then scale to glob
            from math import ceil
            if wsh>ssz[0]: wsh = ssz[0] = ceil(ssz[0]/gsz)*gsz
            if wsw>ssz[1]: wsw = ssz[1] = ceil(ssz[1]/gsz)*gsz
            if ssz!=[h, w]:img = resize(img, ssz)
            if isinstance(mar, float): mar = int(wsz*mar)
            rcs = grid_slice(*ssz, wsh, wsw, mar)
            if len(rcs)>1: info(1, len(rcs))
            rst = f(img[rcs[0]], *p[1:], **fp)
            k = rst.shape[0]/(rcs[0][0].stop - rcs[0][0].start)
            if len(rcs)==1 and ssz!=[h, w]:
                rst = resize(rst, (int(h*k), int(w*k)))
            if len(rcs)==1: return rst if samecore else rst.get()
            def sk(ss, k):
                sr = slice(int(ss[0].start*k), int(ss[0].stop*k))
                sc = slice(int(ss[1].start*k), int(ss[1].stop*k))
                return sr, sc
            outshp = int(img.shape[0]*k), int(img.shape[1]*k)
            outshp = outshp + rst.shape[2:]
            weights = np.zeros(rst.shape[:2], dtype='uint16')
            if rst.ndim==3: weights = weights[:,:,None]
            weights += int(mar * k) + 1
            for i in range(int(mar*k), 0, -1):
                weights[i-1,:] = weights[-i,:] = i
                weights[:,i-1] = weights[:,-i] = i
            buf = np.zeros(outshp, dtype=np.float32)
            count = np.zeros(outshp[:2], dtype='uint16')
            if rst.ndim==3: count = count[:,:,None]
            buf[sk(rcs[0], k)] = rst * weights
            count[sk(rcs[0], k)] += weights
            for i in range(1, len(rcs)):
                info(i+1, len(rcs))
                rst = f(img[rcs[i]], *p[1:], **fp)
                buf[sk(rcs[i], k)] += rst * weights
                count[sk(rcs[i], k)] += weights
            np.divide(buf, count, out=buf, casting='unsafe')
            if ssz!=[h, w]: 
                buf = resize(buf, (int(h*k), int(w*k)))
            rst = buf.astype(rst.dtype)
            return rst if samecore else rst.get()
        return wrap
    return wrapf

if __name__ == '__main__':
    '''
    img = np.zeros((1, 64, 512, 512), dtype=np.float32)
    core = np.zeros((32, 64, 3, 3), dtype=np.float32)
    conv(img, core)
    '''
    import numpy as np
    a = np.arange(4).reshape(1,1,2,2)
    print(a)
    core = np.ones(3).reshape(1,1,1,3)
    print(conv(a, core, mar=(0,1)))
