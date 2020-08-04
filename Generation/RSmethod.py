import numpy as np
import matplotlib.pyplot as plt
import plyfile

# parameters
mm = 1e-3
um = 1e-6
nm = 1e-9

class Propagation:
    def __init__(self, z, plypath, angle=0, pp=3.6 * um, scaleXY=0.03, scaleZ=0.25, w=3840, h=2160):
        self.z = z
        self.pp = pp
        self.scaleXY = scaleXY
        self.scaleZ = scaleZ
        self.width = w
        self.height = h
        self.angle = angle
        self.theta = self.angle * (np.pi / 180)
        with open(plypath, 'rb') as f:
            self.plydata = plyfile.PlyData.read(f)
            self.plydata = np.array(self.plydata.elements[1].data)
        self.wvl_R = 638 * nm
        self.wvl_G = 520 * nm
        self.wvl_B = 450 * nm

    def k(self, wvl):
        return (np.pi *2) / wvl

    def h_RS_re(self, x1, y1, z1, x2, y2, z2, wvl):
        """
        impulse response function of RS propagation method
        """
        r = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1 - z2)**2)
        h_r = np.cos(self.k(wvl) * (r + x1 * np.sin(self.theta) + y1 * np.sin(self.theta)))
        h_i = np.sin(self.k(wvl) * (r + x1 * np.sin(self.theta) + y1 * np.sin(self.theta)))
        return h_r / r, h_i / r

    def anti(self, wvl, z):
        """
        anti aliasing condition
        """
        t = wvl / (2 * self.pp)
        t = (t / np.sqrt(1 - t**2)) * z
        return np.abs(t)

    def RS_phase(self, wvl):
        """
        return phase encoded fringe pattern
        """
        ch = np.zeros((self.height, self.width), dtype='complex64')
        if wvl == self.wvl_G:
            color = 'green'
        elif wvl == self.wvl_B:
            color = 'blue'
        else:
            color = 'red'
        for n in range(len(self.plydata)):
            x0 = self.plydata['x'][n] * self.scaleXY
            y0 = self.plydata['y'][n] * self.scaleXY
            z0 = self.plydata['z'][n] * self.scaleZ
            amp = self.plydata[color][n] * (1/wvl)
            cmap = np.zeros((self.height, self.width), dtype='complex64')
            for i in range(self.height):
                for j in range(self.width):
                    x = (j - self.width / 2) * self.pp
                    y = (i - self.height / 2) * self.pp
                    c1, c2 = self.h_RS_re(x0, y0, z0, x, y, self.z, wvl)
                    c1 = c1 * amp
                    c2 = c2 * amp
                    cmap[i, j] = c1 + 1j * c2
            ch += cmap
            print(color, ' ', n, 'th point done')
        return ch

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
        R = self.RS_phase(self.wvl_R)
        G = self.RS_phase(self.wvl_G)
        B = self.RS_phase(self.wvl_B)
        img = np.zeros((self.height, self.width, 3))
        img[:, :, 0] = self.normalize(R, type)
        img[:, :, 1] = self.normalize(G, type)
        img[:, :, 2] = self.normalize(B, type)
        plt.imsave(fname, img)
        return img

if __name__ == '__main__':
    p = Propagation(1, 'point_3.ply', pp=4.6 * um, w= 1920, h = 1080)
    print(p.plydata.shape)
    print(p.plydata['x'][2])

    ch = p.colorimg('3point.bmp')
    plt.imshow(ch)
    plt.show()