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
    ASM Propagation simulation using multi-core processing
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

    def k(self, wvl):
        return (np.pi *2) / wvl

    def fft(self, f):
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(f)))

    def ifft(self, f):
        return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(f)))

    def asm_kernel(self, zz, wvl):
        """
        ASM kernel
        """
        sizeX = self.pp * self.width
        sizeY = self.pp * self.height
        res = self.width / 2
        a = np.zeros((self.height, self.width), dtype='complex128')
        for i in range(self.height):
            for j in range(self.width):
                fx = (j / self.pp - 0.5) / self.pp / res #(-1 / (2 * self.pp)) + (1 / sizeX) * j
                fy = (i / self.pp - 0.5) / self.pp / res #(1 / (2 * self.pp)) - (1 / sizeY) * (i + 1)
                if np.sqrt(fx ** 2 + fy ** 2) < 1 / wvl:
                    fx = (fx + np.sin(self.thetaX)/wvl) * wvl
                    fy = (fy + np.sin(self.thetaY)/wvl) * wvl
                    a[i, j] = np.exp(1j * self.k(wvl) * zz * np.sqrt(1 - fx ** 2 - fy ** 2))
        return a

    def point_map(self, x1, y1):
        """find object point map"""
        ph = (self.scaleXY * 2) / self.height
        a = np.zeros((self.height, self.width))
        for i in range(self.width):
            for j in range(self.height):
                x = (i - self.width / 2) * ph
                y = (j - self.height / 2) * ph
                if (x - ph < x1 < x + ph) and (y - ph < y1 < y + ph):
                    a[j, i] = 1  # amplitude
                    print(i, ", ", j)
        return a

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
        x0 = self.plydata['x'][n] * self.scaleXY
        y0 = self.plydata['y'][n] * self.scaleXY
        zz = self.z - self.plydata['z'][n] * self.scaleZ
        amp = self.plydata[color][n] * self.point_map(x0, y0)
        amp = self.fft(amp)
        ch = self.ifft(amp * self.asm_kernel(zz, wvl))
        print(n, ' point', color ,' is done')
        return ch

    def FFT_R(self, n):
        return self.ASM_FFT(n, self.wvl_R)

    def FFT_G(self, n):
        return self.ASM_FFT(n, self.wvl_G)

    def FFT_B(self, n):
        return self.ASM_FFT(n, self.wvl_B)

    def multicore(self, wvl):
        """
        using multicore processing
        """
        if wvl == self.wvl_G:
            func = self.FFT_R
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
    p = Propagation(1,'point_3.ply', angleY=20)
    print(p.plydata.shape)
    p.colorimg('ASM_3point_offaxis_2.bmp')
