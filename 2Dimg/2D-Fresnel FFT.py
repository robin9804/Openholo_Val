"""
Fresnel FFT를 활용하여 홀로그램 생성하기 workflow
1. input image RGB로 분리
2. RGB 파장에 따른 이미지 resize (B/R, B/G)로
3. 이미지 가운데로 분류
4. Fresnel FFT 적용 (with zero padding)
5. Phase angle 추출
"""
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from PIL import Image


# parameters
mm = 1e-3
um = mm * mm
nm = um * mm
wvls = [639 * nm, 525 * nm, 463 * nm]
wvl_R = 639 * nm  # Red
wvl_G = 525 * nm  # Green
wvl_B = 463 * nm  # Blue

# SLM parameters
w = 3840  # width
h = 2160  # height
pp = 3.6 * um  # SLM pixel pitch
scaleXY = 0.03
scaleZ = 0.25

# image input
img = Image.open('dice_black.bmp')
img = img.resize((3840, 2160), Image.BILINEAR)
img = np.asarray(img)
img_R = np.double(img[:,:,0])
img_G = np.double(img[:,:,1])
img_B = np.double(img[:,:,2])
w_img = 3840
h_img = 2160

# terms for anti aliasing
Wr = pp * w                 # Receiver plane width

@njit
def zeropadding(wvl):
    ps = wvl / (w_img * pp)  # source plane sampling rate
    return int(Wr / ps + 1)

# resize image
def resizing(img, ratio):
    w2, h2 = new_w(ratio)
    print(w2, ',', h2)
    img_new = np.zeros((h_img, w_img))
    im = Image.fromarray(img)
    im = im.resize((w2, h2), Image.BILINEAR)  # resize image
    im = np.asarray(im)
    print(im.shape)
    img_new[(h_img - h2)//2: (h_img + h2)//2 , (w_img - w2)//2 : (w_img + w2)//2] = im
    return img_new


@njit(nogil=True, cache=True)
def k(wvl):
    return (np.pi * 2) / wvl


#@njit(nogil=True, cache=True)
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

@njit
def Cal(wvl):
    """red = 0, green = 1, blue = 2"""
    wavelength = wvls[wvl]
    image = np.double(img[:, :, wvl])
    image = resizing(image, wvls[2] / wvls[wvl])  # 크롭
    Nzp = zeropadding(wavelength)  # 제로패딩
    ps = wvl / (w_img * pp)  # source plane sampling rate
    ch = np.zeros((Nzp + h, Nzp + w))
    return ps
    
    
def main():
  nzp = zeropadding(wvl_R)
  wr = nzp + w
  hr = nzp + h
  ch = np.zeros((hr, wr))  # resize
  ch[(hr - h)//2: (hr + h)//2 , (wr - w)//2 : (wr + w)//2] = resizing(img_R, wvls[2] / wvls[0])
  pss = wvl_R / (wr * pp)
  psy = wvl_R / (hr * pp)
  ch2 = ch * h_frsn(pss, psy, wr, hr, wvl_R)
  CH = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(ch2)))  # fft
  result = np.zeros((h, w), dtype='complex128')
  result = CH[(hr - h)//2: (hr + h)//2 , (wr - w)//2 : (wr + w)//2]
  result = result * h_frsn(pp, pp, w, h, wvl_R)

  Angle = np.angle(result)

  plt.imshow(Angle)
  ch_real = np.zeros((h, w, 3))
  def normalize(arr):
      arrin = arr - np.min(arr)
      arrin = arrin / np.max(arrin)
      return arrin
  ch_real[:, :, 0] = normalize(Angle)
  ch_real[:, :, 1] = normalize(Angel2)  # GREEN
  ch_real[:, :, 2] = normalize(Angel3)  # BLUE

  plt.imsave('200827Chiamge_aperture3.bmp', ch_real)

  
  
if __name__ == '__main__':
    main()


  
