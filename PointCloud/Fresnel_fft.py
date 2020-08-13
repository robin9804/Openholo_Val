import numpy as np
import plyfile
from math import *
import cmath
import multiprocessing

from encoding import Encoding


class Frsn_FFT(Encoding):
    """Fresnel propagation using FFT and point cloud data"""
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
        self.Nzp = int(self.Wr // self.ps + 1)   # zero padding number
        self.W = int(self.Nzp + self.w)
        self.u_slm = self.h_frsn(self.pp)  # 나중에 여기에 1/(wvl * zz)을 제곱해주면 된다
        self.u_obj = self.h_frsn(self.ps)


    def k(self, wvl):
        return (pi * 2) / wvl

    def pointmap(self, x,  y):
        pmap = np.zeros((self.W, self.W))
        p = int((x+1) * (self.w / 2))
        q = int((1-y) * (self.w / 2))
        pmap[q + self.Nzp // 2, p + self.Nzp // 2] = 1
        print(p, ',', q)
        return pmap

    def refwave(self, wvl, r):
        a = np.zeros((self.h, self.w), dtype='complex128')
        for i in range(self.h):
            for j in range(self.w):
                x = (j - self.w/2) * self.pp
                y = -(i - self.h/2) * self.pp
                a[i, j] = cmath.exp(-1j * self.k(wvl) * (x * sin(self.thetaX + (self.wvl_B-wvl) / self.pp) + y * sin(self.thetaY + (self.wvl_B-wvl) / self.pp)))
        return a / r

    def h_frsn(self, u):
        a = np.zeros((self.W, self.W), dtype='complex128')
        for i in range(self.W):
            for j in range(self.W):
                x = (i - self.W/2) * u
                y = (j - self.W/2) * u
                a[j, i] = cmath.exp(1j * pi * (x*x + y*y))
        return a

    def Frsn_FFT(self, n, wvl):
        """Fresnel FFT kernel"""
        if wvl == self.wvl_G:
            color = 'green'
        elif wvl == self.wvl_B:
            color = 'blue'
        else:
            color = 'red'
        x0 = self.plydata['x'][n]
        y0 = self.plydata['y'][n]
        zz = self.z - self.plydata['z'][n] * self.scaleZ
        amp = self.plydata[color][n]
        u0 = self.pointmap(x0, y0) * (self.u_obj ** (1/(wvl * zz))) * amp
        u0 = self.fft(u0)
        ch = cmath.exp(1j * self.k(wvl) * zz) / (1j * wvl * zz) * (self.u_slm ** (1/(wvl * zz)))
        ch = ch * u0
        ch = ch[(self.W - self.h) // 2: (self.W + self.h) // 2, (self.W - self.w) // 2: (self.W + self.w) // 2]
        ch = ch * self.refwave(wvl, zz)
        print(n, ' point', color, ' is done')
        return ch

    def FFT_R(self, n):
        return self.Frsn_FFT(n, self.wvl_R)

    def FFT_G(self, n):
        return self.Frsn_FFT(n, self.wvl_G)

    def FFT_B(self, n):
        return self.Frsn_FFT(n, self.wvl_B)

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