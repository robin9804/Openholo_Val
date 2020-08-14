import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import multiprocessing

# insert file name
fname = 'point_3.ply'


# parameters
mm = 10e-3
um = mm*mm
nm = um*mm
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM parameters
w = 3840  # width
h = 2160  # height
pp = 3.6 * um  # SLM pixel pitch
scaleXY = 0.03
scaleZ = 0.25
angleX = 0
angleY = 0
f = 1  # field length

thetaX = angleX * (np.pi / 180)
thetaY = angleY * (np.pi / 180) 

# inline functions
def k(wvl):
    return (np.pi * 2) / wvl  # wave number

def fft(f):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))

def ifft(f):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f)))

# get source file
def loadply(fname):
    import plyfile
    with open(fname, 'rb') as f:
        plydata = plyfile.PlyData.read(f)
        plydata = np.array(plydata.elements[1].data)
    del plyfile
    return plydata

def loadimg(fname):
    from PIL import Image
    im = Image.open(fname)
    arr = np.asarray(im)
    del Image
    return arr


# convolution method
@njit(parallel=True)
def Conv(x1, y1, z1, amp, methods, wvl):
    ch_r = np.zeros((h, w))
    ch_i = np.zeros((h, w))
    z2 = f
    for i in prange(h):
        for j in prange(w):
            x2 = (j - w / 2) * pp
            y2 = -(i - h / 2) * pp
            ch_r[i, j], ch_i[i, j] = methods(x1, y1, z1, x2, y2, z2, wvl)
    # print('point done')
    return (ch_r + 1j*ch_i) * amp

@njit
def h_RS(x1, y1, z1, x2, y2, z2, wvl):
    r = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1 - z2)**2)
    t = (wvl * r) / (2 * pp)
    if (x1 - t < x2 < x1 + t) and (y1 - t < y2 < y1 + t):  # anti aliasing condition
        h_r = np.sin(k(wvl) * (r + x2 * np.sin(thetaX) + y2 * np.sin(thetaY)))
        h_i = np.cos(k(wvl) * (r + x2 * np.sin(thetaX) + y2 * np.sin(thetaY)))
    else:
        h_r, h_i = 0
    return h_r / r**2, h_i / r**2

@njit
def h_Frsn(x1, y1, z1, x2, y2, z2, wvl):
    r = ((x1-x2)**2 + (y1-y2)**2) / (2*z)
    t = (wvl * z) / (2 * pp)
    if (x1 - t < x2 < x1 + t) and (y1 - t < y2 < y1 + t):  # anti aliasing
        h_r = np.cos(k(wvl) * (r + x2 * np.sin(thetaX) + y2 * np.sin(thetaY)))
        h_i = np.sin(k(wvl) * (r + x2 * np.sin(thetaX) + y2 * np.sin(thetaY)))
    else:
        h_r, h_i = 0
    return h_r, h_i

def normalize(arrin):
    arrin -= np.min(arrin)
    arrin = arrin / np.max(arrin)
    return arrin


def main():
    print('Start')


if __name__ == "__main__":
    main()
