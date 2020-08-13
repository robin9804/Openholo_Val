import numpy as np
import plyfile
from math import *
import multiprocessing

from encoding import Encoding


class RS(Encoding):
    mm = 1e-3
    um = mm * mm
    nm = um * mm
    def __init__(self, plypath, f=1, angleX=0, angleY=0, pixel_pitch=3.6 * um, scaleXY=0.03, scaleZ=0.25,
                 width=3840, height=2160, wvl_R=638 * nm, wvl_G=520 * nm, wvl_B=450 * nm):
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

    def k(self, wvl):
        return (pi * 2) / wvl

    def h_RS(self, x1, y1, z1, x2, y2, z2, wvl):
        """Impulse Response of R-S propagation"""
        r = sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) + (z1 - z2) * (z1 - z2))
        t = (wvl * r) / (2 * self.pp)  # anti alliasing conditio
        if (x1 - t < x2 < x1 + t) and (y1 - t < y2 < y1 + t):
            h_r = sin(self.k(wvl) * (r + x2 * sin(self.thetaX) + y2 * sin(self.thetaY)))
            h_i = cos(self.k(wvl) * (r + x2 * sin(self.thetaX) + y2 * sin(self.thetaY)))
        else:
            h_r, h_i = 0
        return h_r / (r * r), h_i / (r * r)

    def Conv(self, n, wvl):
        """Convolution"""
        ch = np.zeros((self.h, self.w), dtype='complex128')
        if wvl == self.wvl_G:
            color = 'green'
        elif wvl == self.wvl_B:
            color = 'blue'
        else:
            color = 'red'
        x0 = self.plydata['x'][n] * self.scaleXY
        y0 = self.plydata['y'][n] * self.scaleXY
        z0 = self.plydata['z'][n] * self.scaleZ
        amp = self.plydata[color][n] * (self.z / wvl)
        for i in range(self.h):
            for j in range(self.w):
                x = (j - self.w / 2) * self.pp
                y = -(i - self.h / 2) * self.pp
                c1, c2 = self.h_RS(x0, y0, z0, x, y, self.z, wvl)
                c1 *= amp
                c2 *= amp
                ch[i, j] = c1 + 1j * c2
            if i % (self.h/10) == 0:
                print(n ,'point', (i//(self.h/10)) * 10,'% done')
        print(n, ' th point ', color, ' Done')
        return ch

    def Conv_R(self, n):
        return self.Conv(n, self.wvl_R)

    def Conv_G(self, n):
        return self.Conv(n, self.wvl_G)

    def Conv_B(self, n):
        return self.Conv(n, self.wvl_B)

    def CalHolo(self, wvl):
        """Calculate hologram"""
        if wvl == self.wvl_G:
            func = self.Conv_G
        elif wvl == self.wvl_B:
            func = self.Conv_B
        else:
            func = self.Conv_R
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

        # encoding 과 노말라이즈 부분은 다른 곳에서 따오는 것으로 하자

