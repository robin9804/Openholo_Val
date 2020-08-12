import numpy as np
import plyfile
from math import *
import cmath
import multiprocessing

from encoding import Encoding


class ASM(Encoding):
    """Angular spectrum method using Convolution and point cloud data"""
    mm = 1e-3
    um = mm * mm
    nm = um * mm
    def __init__(self, plypath, f=1, angleX=0, angleY=0, pixel_pitch=3.6 * um, scaleXY=0.03, scaleZ=0.25,
                 width=3840, height=2160, wvl_R=638*nm, wvl_G=520 * nm, wvl_B=450 * nm):
        self.z = f  # Propagation distance
        self.pp = pixel_pitch
        self.scaleXY = scaleXY
        self.scaleZ = scaleZ
        self.w = width
        self.h = height
        self.thetaX = angleX * (pi / 180)
        self.thetaY = angleY * (pi / 180)
        with open(plypath, 'rb') as f:
            self.plydata = plyfile.PlyData.read(f)
            self.plydata = np.array(self.plydata.elements[1].data)
        self.wvl_R = wvl_R
        self.wvl_G = wvl_G
        self.wvl_B = wvl_B
        self.num_cpu = multiprocessing.cpu_count()  # number of CPU
        self.num_point = [i for i in range(len(self.plydata))]
        self.ps = self.scaleXY / self.w     # source plane pixel pitch (sampling rate)
        self.Wr = self.pp * self.w          # Receiver plane width
        self.Ws = self.scaleXY              # Source plane width

    def k(self, wvl):
        return (pi * 2) / wvl

    def Nzp(self, zz, wvl):
        """Number of zero padding"""
        p = (wvl * zz) / sqrt(4 * self.ps * self.ps - wvl * wvl)
        return int((1 / self.ps) * (self.Wr/2 - self.Ws/2 + p) + 1)

    def asm_kernel(self, zz, wvl):
        """ASM kernel"""
        N = self.Nzp(zz, wvl)
        W = N + self.w
        deltaK = (2 * pi) / (W * self.ps)
        a = np.zeros((W, W), dtype='complex128')
        for i in range(W):
            for j in range(W):
                fx = ((i - W / 2) * deltaK) / self.k(wvl)
                fy = ((j - W / 2) * deltaK) / self.k(wvl)
                if np.sqrt(fx * fx + fy * fy) < (1 / wvl) ** 3:
                    a[i, j] = cmath.exp(1j * self.k(wvl) * zz * cmath.sqrt(1 - fx * fx - fy * fy))
        print(zz, 'kernel ready')
        return a

    def refwave(self, wvl, r):
        a = np.zeros((self.h, self.w), dtype='complex128')
        for i in range(self.h):
            for j in range(self.w):
                x = (j - self.w/2) * self.pp
                y = -(i - self.h/2) * self.pp
                a[i, j] = cmath.exp(-1j * self.k(wvl) * (x * sin(self.thetaX) + y * sin(self.thetaY)))
        return a / r

    def ASM_FFT(self, n, wvl):
        """
        ASM FFT kernel
        """
        if wvl == self.wvl_G:
            color = 'green'
        elif wvl == self.wvl_B:
            color = 'blue'
        else:
            color = 'red'
        x0 = self.plydata['x'][n]
        y0 = self.plydata['y'][n]
        zz = self.z - self.plydata['z'][n] * self.scaleZ
        # point map
        N = self.Nzp(zz, wvl)
        W = N + self.w
        pmap = np.zeros((W, W))
        p = np.int((x0 + 1) * (self.w / 2))
        q = np.int((1 - y0) * (self.w / 2))
        pmap[q + N//2, p + N//2] = 1
        print(n, 'th p map done')
        amp = self.plydata[color][n] * pmap
        amp = self.fft(amp)
        ch = self.ifft(amp * self.asm_kernel(zz, wvl))
        ch = ch[(W-self.h)//2: (W+self.h)//2, (W-self.w)//2: (W + self.w)//2]
        ch = ch * self.refwave(wvl, zz)
        print(n, ' point', color,' is done')
        return ch

    def FFT_R(self, n):
        return self.ASM_FFT(n, self.wvl_R)

    def FFT_G(self, n):
        return self.ASM_FFT(n, self.wvl_G)

    def FFT_B(self, n):
        return self.ASM_FFT(n, self.wvl_B)

    def CalHolo(self, wvl):
        """Calculate hologram using multicore processing"""
        if wvl == self.wvl_G:
            func = self.FFT_G
        elif wvl == self.wvl_B:
            func = self.FFT_B
        else:
            func = self.FFT_R
        print(self.num_cpu, " core Ready")
        result = np.zeros((self.h, self.w), dtype='complex128')
        s = np.split(self.num_point, [i * self.num_cpu for i in range(1, len(self.plydata) // self.num_cpu)])
        for n in s:
            pool = multiprocessing.Pool(processes=self.num_cpu)
            summ = pool.map(func, list(n))
            for i in range(len(summ)):
                result += summ[i]
            pool.close()
            pool.join()
        return result