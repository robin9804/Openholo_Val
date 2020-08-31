import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from PIL import Image

from Encode import Encoding

# parameters
mm = 1e-3
um = mm * mm
nm = um * mm
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM size
w = 3840
h = 2160
pp = 3.6 * um  # SLM pixel pitch
sizeXY = 0.03
sizeZ = 0.25

# Input image parameters
h_i = 1080
w_i = 1920
w_new = int(sizeXY / pp + 1)  # image resize
h_new = int(w_new * (9 / 16) + 1)


@njit
def k(wvl):
	return (2 * np.pi) / wvl


@njit
def Nzp(wvl, zz, ps):
	p = (wvl * zz) / np.sqrt(4 * ps**2 - wvl**2)
	return int((1 / ps) * (pp * w / 2 - sizeXY / 2 + p))


nzp = 4000


@njit
def asm_kernel(wvl, zz, nzp):
	ww = int(w_new + nzp)
	hh = int(h_new + nzp)
	h_i = np.zeros((hh, ww))
	h_r = np.zeros((hh, ww))
	delta = 1 / ww / pp  # sampling perieod
	for i in range(ww):
		for j in range(hh):
			fx = (i - ww / 2) * delta
			fy = (j - hh / 2) * delta
			re = np.cos(
			    -2 * np.pi * zz * np.sqrt((1 / wvl)**2 - fx * fx - fy * fy))
			im = np.sin(
			    -2 * np.pi * zz * np.sqrt((1 / wvl)**2 - fx * fx - fy * fy))
			h_r[j, i] = re
			h_i[j, i] = im
	return h_r + 1j * h_i


@njit
def refwave(wvl, wr, hr, thetaX, thetaY):
	a = np.zeros((hr, wr))
	b = np.zeros((hr, wr))
	for i in range(hr):
		for j in range(wr):
			x = (j - wr / 2) * pp
			y = -(i - hr / 2) * pp
			a[i, j] = np.cos(
			    -k(wvl) * (x * np.sin(thetax) + y * np.sin(thetay)))
			b[i, j] = np.sin(
			    -k(wvl) * (x * np.sin(thetax) + y * np.sin(thetay)))
	return a / r - 1j * (b / r)


class AngularSpectrumMethods(Encoding):
	def __init__(self, imgpath, f=1, angleX=0, angleY=0):
		self.zz = f
		self.thetaX = angleX * (np.pi * 180)
		self.thetaY = angleY * (np.pi * 180)
		self.imgin = np.asarray(Image.open(imgpath))
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
		img_new[(h_img - h2) // 2:(h_img + h2) // 2, (w_img - w2) //
		        2:(w_img + w2) // 2] = im
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
		h_r = nzp + h_new  # reciever size and zero padding
		w_r = nzp + w_new
		ch = np.zeros((h_r, w_r))
		im = Image.fromarray(image)
		im = im.resize((w_new, h_new), Image.BILINEAR)
		im = np.asarray(im)
		ch[nzp // 2:nzp // 2 + h_new, nzp // 2:nzp // 2 + w_new] = im
		# ASM part
		CH = self.fft(ch)
		A = asm_kernel(wvl, self.zz, nzp)
		CH *= A
		HO = self.ifft(CH)
		# Reference wave
		HO += refwave(wvl, w_r, h_r, self.thetaX, self.thetaY) * 4  # amp
		result = np.zeros((h, w), dtype='complex128')
		result = HO[(h_r - h) // 2:(h_r + h) // 2, (w_r - w) // 2:(w_r + w) //
		            2]  # crop
		return result
