import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from depthmap.encoding import Encoding
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from numba import njit

# parameters
mm = 1e-3
um = mm * mm
nm = um * mm
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM parameters
w = 1920  # width
h = 1080  # height
pp = 3.6 * um  # SLM pixel pitch
scaleXY = 0.03
scaleZ = 0.25

fname = 'original.bmp'

@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl

@njit(nogil=True)
def h_RS(x1, y1, z1, x2, y2, z2, wvl):
    """Impulse Response of R-S propagation"""
    r = np.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2))
    t = (wvl * r) / (2 * pp)  # anti alliasing conditio
    if (x1 - t < x2 < x1 + t) and (y1 - t < y2 < y1 + t):
        h_r = np.sin(k(wvl) * r)
        h_i = np.cos(k(wvl) * r)
    else:
        h_r = 0
        h_i = 0
    return h_r / (r * r), h_i / (r * r)

@njit(nogil=True)
def pointConv(x1, y1, z, wvl, amp):
    u_re = np.zeros((h,w))
    u_im = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            x2 = (j - w / 2) * pp
            y2 = -(i - h / 2) * pp
            re, im = h_RS(x1, y1, 0, x2, y2, z, wvl)
            re = re * amp
            im = im * amp
            u_re[i, j] = re
            u_im[i, j] = im
    # print(x1, ', ', y1, 'done')
    return u_re, u_im


class RS(Encoding):
    def __init__(self, imgpath, f=1):
        self.z = f  # Propagation distance
        self.imagein = np.asarray(Image.open(imgpath))
        self.num_cpu = multiprocessing.cpu_count()  # number of CPU
        self.imgR = np.double(self.imagein[:, :, 0])
        self.imgG = np.double(self.imagein[:, :, 1])
        self.imgB = np.double(self.imagein[:, :, 2])
        self.num_point = [i for i in range(w)]

    def rowConv(self,image, row, wvl):
        u_re = np.zeros((h, w))
        u_im = np.zeros((h, w))
        for n in range(h):
            if image[n, row] == 0:
                continue
            amp = image[n, row]
            x1 = (row - w / 2) * pp
            y1 = (n - h / 2) * pp
            re, im = pointConv(x1, y1, 0, amp, wvl)
            u_re += re
            u_im += im
            print(n, ' raw ready')
        return u_re + 1j * u_im

    def Cal(self, row, color='red'):
        """Convolution"""
        if color == 'green':
            wvl = wvl_G
            img = self.imgG
        elif color == 'blue':
            wvl = wvl_B
            img = self.imgB
        else:
            wvl = wvl_R
            img = self.imgR
        ch = self.rowConv(img, row, wvl)
        print(row, ' th point ', color, ' Done')
        return ch

    def Conv_R(self, n):
        return self.Cal(n, 'red')

    def Conv_G(self, n):
        return self.Cal(n, 'green')

    def Conv_B(self, n):
        return self.Cal(n, 'blue')

    def CalHolo(self, color='red'):
        """Calculate hologram"""
        if color == 'green':
            func = self.Conv_G
        elif color == 'blue':
            func = self.Conv_B
        else:
            func = self.Conv_R
        print(self.num_cpu, " core Ready")
        ch = np.zeros((h, w), dtype='complex128')
        count = np.split(self.num_point, [i * self.num_cpu for i in range(1, w // self.num_cpu)])
        print(count)
        for n in count:
            with ProcessPoolExecutor(self.num_cpu) as ex:
                cache = [result for result in ex.map(func, list(n))]
                cache = np.asarray(cache)
                print(n, 'steps done')
                for j in range(len(n)):
                    ch += cache[j, :, :]
        return ch
