import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from depthmap.encoding import Encoding
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
scaleXY = 0.03      # Source plane width
scaleZ = 0.25

Ws = scaleXY
Wr = pp * w         # Reciver plane width
ps = scaleXY / w

# zero padding number
nzp = int(Wr / ps + 1)
h_r = int(h + nzp)
w_r = int(w + nzp)


@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl


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
    return re + 1j * im


@njit(nogil=True)
def Refwave(wvl, r, thetax, thetay):
    a = np.zeros((h, w))
    b = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            x = (j - w / 2) * pp
            y = -(i - h / 2) * pp
            a[i, j] = np.cos(k(wvl) * (x * np.sin(thetax) + y * np.sin(thetay)))
            b[i, j] = np.sin(k(wvl) * (x * np.sin(thetax) + y * np.sin(thetay)))
    return a / r + 1j * (b / r)


class Frsn(Encoding):
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
        img_new = np.zeros((h_r, w_r))
        im = Image.fromarray(img)
        im = im.resize((w_n, h_n), Image.BILINEAR)  # resize image
        im = np.asarray(im)
        print(im.shape)
        img_new[(h_r - h_n) // 2:(h_r + h_n) // 2, (w_r - w_n) // 2:(w_r + w_n) // 2] = im
        return img_new

    def Cal(self, color):
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
        ratio = wvl / wvl_B
        zzz = ratio * self.zz
        psx = (wvl * self.zz) / (w_r * pp)
        psy = (wvl * self.zz) / (h_r * pp)
        self.ch2 = image * h_frsn(ps, ps, w_r, h_r, zzz, wvl)
        CH1 = self.fft(self.ch2)
        result = CH1 * h_frsn(pp, pp, w_r, h_r, zzz, wvl) * (np.exp(1j * k(wvl) * zzz) / (1j * wvl * zzz))
        result = result[(h_r - h) // 2: (h_r + h) // 2, (w_r - w) // 2: (w_r + w) // 2]
        result *= Refwave(wvl, zzz, self.thetaX, self.thetaY)
        return result
