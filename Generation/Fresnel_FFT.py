import numpy as np
import matplotlib.pyplot as plt
import plyfile
import multiprocessing

# parameters
mm = 1e-3
um = 1e-6
nm = 1e-9


class Propagation:
    """
    Fresnel FFT Propagation simulation using multi-core processing
    """
    def __init__(self, z, plypath, angleX=0, angleY=0, pp=3.6 * um, scaleXY=0.03, scaleZ=0.25, w=3840, h=2160):
        self.z = z
        self.pp = pp
        self.scaleXY = scaleXY
        self.scaleZ = scaleZ
        self.width = w
        self.height = h
        self.angleX = angleX
        self.angleY = angleY
        self.thetaX = self.angleX * (np.pi / 180)
        self.thetaY = self.angleY * (np.pi / 180)
        with open(plypath, 'rb') as f:
            self.plydata = plyfile.PlyData.read(f)
            self.plydata = np.array(self.plydata.elements[1].data)
        self.wvl_R = 638 * nm
        self.wvl_G = 520 * nm
        self.wvl_B = 450 * nm
        self.num_cpu = multiprocessing.cpu_count()
        self.num_point = [i for i in range(len(self.plydata))]
        self.ph = (self.scaleXY * 2) / self.height

    def k(self, wvl):
        return (np.pi *2) / wvl

    def fft(self, f):
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))

    def ifft(self, f):
        return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f)))

    def point_map(self, x1, y1):
        """find object point map"""
        a = np.zeros((self.height, self.width))
        for i in range(self.width):
            for j in range(self.height):
                x = (i - self.width / 2) / (self.height / 2) # * self.ph
                y = -(j - self.height / 2)/ (self.height / 2) #* self.ph
                if (x - self.ph < x1 < x + self.ph) and (y - self.ph < y1 < y + self.ph):
                    a[j, i] = 1  # amplitude
                    print(i, ", ", j)
        return a

    def h_frsn(self, zz, wvl, p):
        a = np.zeros((self.height, self.width), dtype='complex128')
        for i in range(self.width):
            for j in range(self.height):
                x = (i - self.width / 2) * p
                y = (j - self.height / 2) * p
                a[j, i] = np.exp(1j * self.k(wvl) * (x ** 2 + y ** 2) / (2 * zz))
        return a

    def Frsn_FFT(self, n, wvl):
        """
        Fresnel FFT kernel
        """
        if wvl == self.wvl_G:
            color = 'green'
        elif wvl == self.wvl_B:
            color = 'blue'
        else:
            color = 'red'
        x0 = self.plydata['x'][n] * self.scaleXY
        y0 = self.plydata['y'][n] * self.scaleXY
        zz = self.z - self.plydata['z'][n] * self.scaleZ
        amp = self.plydata[color][n]
        u0 = self.point_map(x0, y0) * self.h_frsn(zz, wvl, self.ph) * amp
        u0 = self.fft(u0)
        ch = np.exp(1j * self.k(wvl) * zz) / (1j * wvl * zz)  # +  x2 * np.sin(self.thetaX) + y2 * np.sin(self.thetaY))
        ch = ch * self.h_frsn(zz, wvl, self.pp)
        print(n, ' point', color, ' is done')
        return ch * u0

    def FFT_R(self, n):
        return self.Frsn_FFT(n, self.wvl_R)

    def FFT_G(self, n):
        return self.Frsn_FFT(n, self.wvl_G)

    def FFT_B(self, n):
        return self.Frsn_FFT(n, self.wvl_B)

    def multicore(self, wvl):
        """
        using multicore processing
        """
        if wvl == self.wvl_G:
            func = self.FFT_G
        elif wvl == self.wvl_B:
            func = self.FFT_B
        else:
            func = self.FFT_R
        print(self.num_cpu, " core Ready")
        result = np.zeros((self.height, self.width), dtype='complex128')
        s = np.split(self.num_point, [i * self.num_cpu for i in range(1, len(self.plydata) // self.num_cpu)])
        for n in s:
            pool = multiprocessing.Pool(processes=self.num_cpu)
            summ = pool.map(func, list(n))
            for i in range(len(summ)):
                result += summ[i]
            pool.close()
            pool.join()
        return result

    def normalize(self, arr, type='phase'):
        if type == 'phase':
            arrin = np.imag(arr)
        elif type == 'real':
            arrin = np.real(arr)
        elif type == 'angle':
            arrin = np.angle(arr)
        elif type == 'amplitude':
            arrin = np.abs(arr)
        arrin -= np.min(arrin)
        arrin = arrin / np.max(arrin)
        return arrin

    def colorimg(self, fname, type='phase'):
        """make color image"""
        R = self.multicore(self.wvl_R)
        G = self.multicore(self.wvl_G)
        B = self.multicore(self.wvl_B)
        img = np.zeros((self.height, self.width, 3))
        img[:, :, 0] = self.normalize(R, type)
        img[:, :, 1] = self.normalize(G, type)
        img[:, :, 2] = self.normalize(B, type)
        plt.imsave(fname, img)
        return img

    def singlechannel(self, fname, wvl):
        """
        extract single channel img
        """
        img = self.multicore(wvl)
        phaseimg = self.normalize(img, 'phase')
        f_phase = 'IM' + fname
        plt.imsave(f_phase, phaseimg, cmap='gray')
        realimg = self.normalize(img, 'real')
        f_real = 'RE' + fname
        plt.imsave(f_real, realimg, cmap='gray')
        return phaseimg, realimg

if __name__ == '__main__':
    p = Propagation(1, 'point_3.ply', angleY=20)
    print(p.plydata.shape)
    p.colorimg('FresnelFFT_3point_offaxis.bmp')
