import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from depthmap.encoding import Encoding
from concurrent.futures import ProcessPoolExecutor
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
w_img = int(scaleXY / pp)   # image resize (sampling period)
h_img = int(w_img * (9 / 16))
Wr = pp * w         # Reciver plane width


# zero padding
@njit
def zeropadding(wvl):
    ps = wvl / (w_img * pp)  # source plane sampling rate
    return int(Wr / ps + 1)
nzp = w         # w 만큼 zero padidng 시켜보자


@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl


@njit(nogil=True, cache=True)
def h_frsn(pixel_pitch_x, pixel_pitch_y, nx, ny, wvl):
    re = np.zeros((ny, nx))
    im = np.zeros((ny, nx))
    for i in range(nx):
        for j in range(ny):
            x = (i - nx / 2) * pixel_pitch_x
            y = (j - ny / 2) * pixel_pitch_y
            re[j, i] = np.cos((np.pi / wvl) * (x * x + y * y))
            im[j, i] = np.sin((np.pi / wvl) * (x * x + y * y))
    return re + 1j * im


@njit(nogil=True)
def Refwave(wvl, r, thetax, thetay):
    a = np.zeros((h, w))
    b = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            x = (j - w / 2) * pp
            y = -(i - h / 2) * pp
            a[i, j] = np.cos(-k(wvl) * (x * np.sin(thetax) + y * np.sin(thetay)))
            b[i, j] = np.sin(-k(wvl) * (x * np.sin(thetax) + y * np.sin(thetay)))
    return a / r - 1j * (b / r)


class Frsn(Encoding):
    def __init__(self, imgpath, f=1, angleX=0, angleY=0):
        self.zz = f
        self.imagein = np.asarray(Image.open(imgpath))
        self.num_cpu = 16 #multiprocessing.cpu_count()  # number of CPU
        self.thetaX = angleX * (np.pi / 180)
        self.thetaY = angleY * (np.pi / 180)
        self.img_R = self.resizeimg(wvl_R, self.imagein[:, :, 0])
        self.img_G = self.resizeimg(wvl_G, self.imagein[:, :, 1])
        self.img_B = np.double(self.imagein[:, :, 2])

    def resizeimg(self, wvl, img):
        w_new = int((wvl_B / wvl) * w + 1)
        h_new = int((wvl_B / wvl) * h + 1)
        img_new = np.zeros((h_img, w_img))
        im = Image.fromarray(img)
        im = im.resize((w2, h2), Image.BILINEAR)  # resize image
        im = np.asarray(im)
        print(im.shape)
        img_new[(h_img - h2)//2: (h_img + h2)//2 , (w_img - w2)//2 : (w_img + w2)//2] = im
        return img_new

    def Cal(self, wvl):
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
        h_r = nzp + h_new   # reciever size
        w_r = nzp + w_new
        ch = np.zeros((h_r, w_r))
        im = Image.fromarray(image)
        im = im.resize((w_new, h_new), Image.BILINEAR)
        im = np.asarray(im)
        ch[nzp//2:nzp//2+h_new,nzp//2:nzp//2+w_new] = im
        psx = wvl / ((w_r) * pp)
        psy = wvl / ((h_r) * pp)
        ch2 = ch * h_frsn(psx, psy, w_r, h_r, wvl)
        CH1 = self.fft(ch2)
        result = CH1[(h_r - h)//2: (h_r + h)//2 , (w_r - w)//2 : (w_r + w)//2]
        result *= h_frsn(pp, pp, w, h, wvl)
        return result
        
