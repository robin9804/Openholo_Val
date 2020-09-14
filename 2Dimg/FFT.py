import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from openholo.funcs.encoding import Encoding
from numba import njit

# parameters
mm = 1e-3
um = mm * mm
nm = um * mm
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM parameters
w = 3840            # width
h = 2160            # height
pp = 3.6 * um       # SLM pixel pitch
nzp = w

scaleX = 0.03

@njit
def k(wvl):
    return (np.pi * 2) / wvl


@njit
def limits(u, z, wvl):
    # u is delta u
    return (1/wvl) * np.sqrt((2 * u * z)**2 + 1)


@njit
def asm_kernel(wvl, z):
    deltax = 1 / (w * pp * 4)     # sampling period
    deltay = 1 / (h * pp * 4)
    a = np.zeros((h*2, w*2))        # real part
    b = np.zeros((h*2, w*2))        # imaginary part
    for i in range(w*2):
        for j in range(h*2):
            fx = ((i - w) * deltax)
            fy = ((j - h) * deltay)
            delx = limits(fx, z, wvl)
            dely = limits(fy, z, wvl)
            if -delx < fx < delx and -dely < fy < dely:
                #(fx * fx + fy * fy) < (1 / (wvl * wvl)):
                a[j, i] = np.cos(2 * np.pi * z * np.sqrt((1/wvl)**2 - fx * fx - fy * fy))
                b[j, i] = np.sin(2 * np.pi * z * np.sqrt((1/wvl)**2 - fx * fx - fy * fy))
    print(z, 'kernel ready')
    return a + 1j * b


@njit(nogil=True, cache=True)
def h_frsn(pixel_pitch_x, pixel_pitch_y, nx, ny, zz, wvl):
    re = np.zeros((ny, nx))
    im = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            x = (i - nx / 2) * pixel_pitch_x
            y = (j - ny / 2) * pixel_pitch_y
            re[j, i] = np.cos((np.pi / (wvl * zz)) * (x * x + y * y))
            im[j, i] = np.sin((np.pi / (wvl * zz)) * (x * x + y * y))
    return re - 1j * im


class FFT(Encoding):
    def __init__(self, imgpath, f=1, angleX=0, angleY=0):
        self.zz = f
        self.imagein = np.asarray(Image.open(imgpath))
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        self.img_R = np.double(self.resizeimg(wvl_R, self.imagein[:, :, 0])) / 255
        self.img_G = np.double(self.resizeimg(wvl_G, self.imagein[:, :, 1])) / 255
        self.img_B = np.double(self.resizeimg(wvl_B, self.imagein[:, :, 2])) / 255

    def resizeimg(self, wvl, img):
        """RGB 파장에 맞게 원본 이미지를 리사이징 + zero padding"""
        w_n = int(w * (wvl_B / wvl))
        h_n = int(h * (wvl_B / wvl))
        img_new = np.zeros((h*2, w*2))
        im = Image.fromarray(img)
        im = im.resize((w_n, h_n), Image.BILINEAR)  # resize image
        im = np.asarray(im)
        print(im.shape)
        img_new[(h*2 - h_n) // 2:(h*2 + h_n) // 2, (w*2 - w_n) // 2:(w*2 + w_n) // 2] = im
        return img_new

    def Frsn(self, color):
        if color == 'green':
            wvl = wvl_G
            image = self.img_G
        elif color == 'blue':
            wvl = wvl_B
            image = self.img_B
        else:
            wvl = wvl_R
            image = self.img_R
        # resize image
        zzz = 1 * self.zz
        ps = scaleX / w
        #psx = (wvl * self.zz) / (w * 10 * pp)
        #psy = (wvl * self.zz) / (10 * h * pp)
        self.ch2 = image * h_frsn(ps, ps, w + w, h + h, zzz, wvl)
        CH1 = np.fft.fft2(np.fft.fftshift(self.ch2))
        result = CH1 * -h_frsn(pp, pp, w+w, h+h, zzz, wvl)
        result = result[h // 2: (3*h) // 2, w // 2: (3*w) // 2]
        #result *= Refwave(wvl, zzz, self.thetaX, self.thetaY)
        return result

    def ASM(self, color):
        if color == 'green':
            wvl = wvl_G
            image = self.img_G
        elif color == 'blue':
            wvl = wvl_B
            image = self.img_B
        else:
            wvl = wvl_R
            image = self.img_R
        CH = self.fft(image)
        CH = CH * asm_kernel(wvl, self.zz)
        result = self.ifft(CH)
        result = result[h // 2: (3 * h) // 2, w // 2: (3 * w) // 2]
        return result
