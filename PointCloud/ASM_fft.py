import numpy as np
import plyfile
import multiprocessing

from concurrent.futures import ProcessPoolExecutor
from numba import njit
from encoding import Encoding

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
ps = scaleXY / w    # source plane pixel pitch (sampling rate)
Wr = pp * w         # Receiver plane width
Ws = scaleXY        # Source plane width

# inline function
@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl

@njit(nogil=True, cache=True)
def nzp(zz, wvl):
    p = (wvl * zz) / np.sqrt(4 * ps * ps - wvl * wvl)
    return int((1 / ps) * (Wr/2 - Ws/2 + p) + 1)

@njit(nogil=True)
def asm_kernel(zz, wvl):
    N = nzp(zz, wvl)
    W = N + w
    deltak = wvl / (W * ps)
    re = np.zeros((W,W))
    im = np.zeros((W,W))
    for i in range(W):
        for j in range(W):
            fx = ((i - W / 2) * deltak)
            fy = ((j - W / 2) * deltak)
            if np.sqrt(fx * fx + fy * fy) < (1 / wvl) ** 3:
                re[j, i] = np.cos(k(wvl) * zz * np.sqrt(1 - fx * fx - fy * fy))
                im[j, i] = np.sin(k(wvl) * zz * np.sqrt(1 - fx * fx - fy * fy))
        print(zz, 'kernel ready')
        return re, im

class ASM(Encoding):
    """Angular spectrum method FFT propagation"""
    def __init__(self, plypath, f=1, angleX=0, angleY=0):
        self.z = f  # Propagation distance
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        with open(plypath, 'rb') as f:
            self.plydata = plyfile.PlyData.read(f)
            self.plydata = np.array(self.plydata.elements[1].data)
        self.num_cpu = multiprocessing.cpu_count()  # number of CPU
        self.num_point = [i for i in range(len(self.plydata))]

    def refwave(self, wvl, r):
        a = np.zeros((self.h, self.w), dtype='complex128')
        for i in range(self.h):
            for j in range(self.w):
                x = (j - self.w/2) * self.pp
                y = -(i - self.h/2) * self.pp
                a[i, j] = np.exp(-1j * self.k(wvl) * (x * np.sin(self.thetaX) + y * np.sin(self.thetaY)))
        return a / r

    def Cal(self, n, color='red'):
        """FFT calcuation"""
        ch = np.zeros((W, W), dtype='complex128')
        if color == 'green':
            wvl = wvl_G
        elif color == 'blue':
            wvl = wvl_B
        else:
            wvl = wvl_R
        x0 = self.plydata['x'][n]
        y0 = self.plydata['y'][n]
        zz = self.z - self.plydata['z'][n] * scaleZ
        # point map
        N = nzp(zz, wvl)
        W = N + w
        pmap = np.zeros((W, W))
        p = np.int((x0 + 1) * (w / 2))
        q = np.int((1 - y0) * (w / 2))
        pmap[q + N//2, p + N//2] = 1
        print(n, 'th p map done')
        amp = self.plydata[color][n] * pmap
        amp = self.fft(amp)
        ch = self.ifft(amp *asm_kernel(zz, wvl))
        ch = ch[(W-h)//2: (W+h)//2, (W-w)//2: (W+w)//2]
        ch = ch * self.refwave(wvl, zz)
        print(n, ' point', color,' is done')

    def FFT_R(self, n):
        return self.Cal(n, 'red')

    def FFT_G(self, n):
        return self.Cal(n, 'green')

    def FFT_B(self, n):
        return self.Cal(n, 'blue')

    def CalHolo(self, color='red'):
        """Calculate hologram"""
        if color == 'green':
            func = self.FFT_G
        elif color == 'blue':
            func = self.FFT_B
        else:
            func = self.FFT_R
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
