import numpy as np
import plyfile
from numba import njit
from concurrent.futures import ProcessPoolExecutor
from functions.encoding import Encoding
import multiprocessing


# parameters
mm = 1e-3
um = mm * mm
nm = um * mm
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM parameters
w = 3840  # width
h = 2160  # height
pp = 3.6 * um  # SLM pixel pitch
scaleXY = 0.03
scaleZ = 0.25


@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl


@njit(nogil=True)
def h_Fresnel(x1, y1, x2, y2, z, wvl):
    """impulse response function of Fresnel propagation method"""
    r = ((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)) / (2*z)
    # t = (wvl * z) / (2 * pp) 안티엘리어싱 무시
    h_r = np.cos(k(wvl) * r)
    h_i = np.sin(k(wvl) * r)
    return h_r, h_i


@njit(nogil=True, cache=True)
def Refwave(wvl, r, thetaX, thetaY):
    """Reference wave"""
    a = np.zeros((h, w))
    b = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            x = (j - w/2) * pp
            y = -(i - h/2) * pp
            a[i, j] = np.cos(k(wvl) * (x * np.sin(thetaX) + y * np.sin(thetaY)))
            b[i, j] = np.sin(k(wvl) * (x * np.sin(thetaX) + y * np.sin(thetaY)))
    return a / r,  b / r


@njit(nogil=True)
def Conv(x1, y1, z1, z2, amp, wvl):
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
    print('point done')
    return (ch_r + 1j * ch_i) * amp


class Frsn(Encoding):
    def __init__(self, plypath, f=1, angleX=0, angleY=0):
        self.z = f  # Propagation distance
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        with open(plypath, 'rb') as f:
            self.plydata = plyfile.PlyData.read(f)
            self.plydata = np.array(self.plydata.elements[1].data)
        self.num_cpu = multiprocessing.cpu_count()  # number of CPU
        self.num_point = [i for i in range(len(self.plydata))]

    def Cal(self, n, color='red'):
        """Convolution"""
        ch = np.zeros((h, w), dtype='complex128')
        if color == 'green':
            wvl = wvl_G
        elif color == 'blue':
            wvl = wvl_B
        else:
            wvl = wvl_R
        x0 = self.plydata['x'][n] * scaleXY
        y0 = self.plydata['y'][n] * scaleXY
        z0 = self.plydata['z'][n] * scaleZ
        amp = self.plydata[color][n] * (self.z / wvl)
        ch = Conv(x0, y0, z0, self.z, amp, wvl)
        print(n, ' th point ', color, ' Done')
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
        count = np.split(self.num_point, [i * self.num_cpu for i in range(1, len(self.plydata) // self.num_cpu)])
        print(count)
        for n in count:
            with ProcessPoolExecutor(self.num_cpu) as ex:
                cache = [result for result in ex.map(func, list(n))]
                cache = np.asarray(cache)
                print(n, 'steps done')
                for j in range(len(n)):
                    ch += cache[j, :, :]
        return ch
