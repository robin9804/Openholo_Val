import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from PIL import Image

# parameters
mm = 1e-3
um = mm * mm
nm = um * mm
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM parameters
w = 3840  # SLM width
h = 2160
pp = 3.6 * um  # SLM pixel pitch
scaleXY = 0.03
scaleZ = 0.25
ps = scaleXY / w_s

# read iamge
img = np.asarray(Image.open('aperture2.bmp'))
img_R = np.double(img[:,:,0])
img_G = np.double(img[:,:,1])
img_B = np.double(img[:,:,2])

w_s = img.shape[1]  # source width
h_s = img.shape[0]  # height


@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl


@njit(nogil=True)
def h_Fresnel(x1, y1, x2, y2, z, wvl):
    """impulse response function of Fresnel propagation method"""
    r = ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)) / (2*z)
    t = (wvl * z) / (2 * pp)
    if (x1 - t < x2 < x1 + t) and (y1 - t < y2 < y1 + t):  # anti aliasing
        h_r = np.cos(k(wvl) * r)
        h_i = np.sin(k(wvl) * r)
    else:
        h_r = 0
        h_i = 0
    return h_r, h_i


@njit(nogil=True)
def pointConv(x1, y1, z1, z2, amp, wvl):
    zz = z2 - z1
    ch_r = np.zeros((h, w))
    ch_i = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            x2 = (j - w / 2) * pp
            y2 = -(i - h / 2) * pp
            re, im = h_Fresnel(x1, y1, x2, y2, zz, wvl)
            ch_r[i, j] = re
            ch_i[i, j] = im
    # print('point done')
    return ch_r * amp,  ch_i * amp


@njit(nogil=True)
def FullConv(image, z, wvl):
    ch_r = np.zeros((h, w))
    ch_i = np.zeros((h, w))
    for p in range(h_s):
        for q in range(w_s):
            if image[p, q] == 0:
                continue
            x1 = (p - w / 2) * ps
            y1 = -(q - h / 2) * ps
            amp = image[p, q]
            u_re, u_im = pointConv(x1, y1, 0, z, amp, wvl)
            print(p,', ', q, " th point done")
            ch_r += u_re
            ch_i += u_im
    print('all point done')
    return ch_r + 1j * ch_r


def normalize(arr):
    arrin = np.copy(np.angle(arr))
    arrin -= np.min(arrin)
    arrin = arrin / (2 * np.pi)
    return arrin


def main():
    import time
    start = time.time()
    print(img_R.shape)
    ch = np.zeros((h, w, 3), dtype='complex128')
    r = Conv(img_R, 1, wvl_R)  # 1점에 0.7 초 걸림
    g = Conv(img_G, 1, wvl_G)
    b = Conv(img_B, 1, wvl_B)

    ch[:, :, 0] = normalize(r)
    ch[:, :, 1] = normalize(g)
    ch[:, :, 2] = normalize(b)

    print(time.time() - start, ' is time')
    plt.imshow(np.real(ch))
    plt.imsave('diceimage2.bmp', ch)

if __name__ == '__main__':
    main()
