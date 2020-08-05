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
    Fresnel Propagation simulation using multi-core processing
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

    def h_Fresnel(self, x1, y1, z1, x2, y2, z2, wvl):
        """
        impulse response function of Fresnel propagation method
        """
        r = ((x1-x2)**2 + (y1-y2)**2) / (2*(z2 - z1))
        t = (wvl * (z2 - z1)) / (2 * self.pp)
        if (x1 - t < x2 < x1 + t) and (y1 - t < y2 < y1 + t):  # anti aliasing
            h:complex = np.exp(1j * self.k(wvl) * (r + x2 * np.sin(self.thetaX) + y2 * np.sin(self.thetaY)))
        else:
            h = 0
        return h

    def Conv(self, n, wvl):
        """
        convolution method with 1 point
        """
        ch = np.zeros((self.height, self.width), dtype='complex128')
        if wvl == self.wvl_G:
            color = 'green'
        elif wvl == self.wvl_B:
            color = 'blue'
        else:
            color = 'red'
        x0 = self.plydata['x'][n] * self.scaleXY
        y0 = self.plydata['y'][n] * self.scaleXY
        z0 = self.plydata['z'][n] * self.scaleZ
        amp = self.plydata[color][n] * (np.exp(1j * self.k(wvl) * (self.z - z0)) / (1j * wvl * (self.z - z0)))
        for i in range(self.height):
            for j in range(self.width):
                x = (j - self.width / 2) * self.pp
                y = (i - self.height / 2) * self.pp
                c = self.h_Fresnel(x0, y0, z0, x, y, self.z, wvl)
                c = c * amp
                ch[i, j] = c
        print(n, " th point ", color, " done")
        return ch

    def Conv_R(self,n):
        return self.Conv(n, self.wvl_R)

    def Conv_G(self,n):
        return self.Conv(n, self.wvl_G)

    def Conv_B(self,n):
        return self.Conv(n, self.wvl_B)

    def multicore(self, wvl):
        """
        using multicore processing
        """
        if wvl == self.wvl_G:
            func = self.Conv_G
        elif wvl == self.wvl_B:
            func = self.Conv_B
        else:
            func = self.Conv_R
        print(self.num_cpu, " core Ready")
        result = np.zeros((self.height, self.width), dtype='complex128')
        s = np.split(self.num_point, [i*self.num_cpu for i in range(1, len(self.plydata) // self.num_cpu)])
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
    # set class
    p = Propagation(1, 'point_3.ply')
    print(p.plydata.shape)

    # get single channel image
    ch1, ch2 = p.singlechannel('fresnel_3p.bmp', p.wvl_G)

    # get color image
    ch = p.colorimg('3p_Frsn_1.bmp')
    plt.imshow(ch)
    plt.show()