import cupy as np
from time import time


def neighbors(shape, core, offset=0, dilation=1):
    shp = [slice(0, i) for i in core]
    idx = np.mgrid[tuple(shp)]
    idx = idx.reshape((len(core), -1))
    # print(core, type(offset), offset)
    offset = (np.array(core)-1)//2*np.array(offset)
    idx -= offset.reshape((-1, 1))

    idx.T[:] *= np.array(dilation)
    acc = np.cumprod((1,)+shape[::-1][:-1])
    return np.dot(idx.T, acc[::-1])


def conv(img, core, group=1, stride=(1, 1), dilation=(1, 1), buf=['']):
    (strh, strw), (dh, dw) = stride, dilation
    n, c, h, w = core.shape
    cimg_w = c * h * w * group
    ni, ci, hi, wi = img.shape
    cimg_h = ni*(hi//strh)*(wi//strw)
    shp = ((0, 0), (0, 0), (h*dh//2, h*dh//2), (w*dw//2, w*dw//2))
    img = np.pad(img, shp, 'constant', constant_values=0)

    acc = np.cumprod((1,)+img.shape[::-1][:-1])
    nbs = neighbors(img.shape[1:], core.shape[1:], (0, 1, 1), (1, dh, dw))

    slicew = np.arange(w*dw//2, w*dw//2+wi, strw) * acc[0]
    sliceh = np.arange(h*dh//2, h*dh//2+hi, strh) * acc[1]
    slicec = np.arange(0, ci, c) * acc[2]
    idx = sliceh.reshape((-1, 1)) + slicew.ravel()
    idx = slicec.reshape((-1, 1)) + idx.ravel()
    idx = idx.reshape((-1, 1)) + nbs
    col_img = img.reshape((ni,-1))[:,idx.ravel()]

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


def pool(img, f, core=(2, 2), stride=(2, 2)):
    (n, c, h, w), (ch, cw), (strh, strw) = img.shape, core, stride
    shp = ((0, 0), (0, 0), ((ch-1)//2,)*2, ((cw-1)//2,)*2)

    pdimg = np.pad(img, shp, 'constant', constant_values=0)

    imn, ic, ih, iw = pdimg.shape
    acc = np.cumprod((1,)+pdimg.shape[::-1][:-1])
    nbs = neighbors(pdimg.shape[2:], core, (1, 1))
    slicew = np.arange((ch-1)//2, iw-((ch-1)//2), strw) * acc[0]
    sliceh = np.arange((cw-1)//2, ih-((cw-1)//2), strh) * acc[1]

    idx = sliceh.reshape((-1, 1)) + slicew.ravel()
    idx = idx.reshape((-1, 1)) + nbs
    colimg = pdimg.reshape((imn*ic, -1))[:, idx.ravel()]

    colimg = colimg.reshape((-1, len(nbs)))
    if f == 'max':
        colimg = colimg.max(axis=1)
    if f == 'mean':
        colimg = colimg.mean(axis=1)
    return colimg.reshape((n, c, h//strh, w//strw))


def maxpool(i, c=(2, 2), s=(2, 2)): return pool(i, 'max', c, s)


def avgpool(i, c=(2, 2), s=(2, 2)): return pool(i, 'mean', c, s)


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


if __name__ == '__main__':
    pass
