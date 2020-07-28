import numpy as np
import matplotlib.pyplot as plt
import plyfile

# parameters
mm = 1e-3
um = 1e-6
nm = 1e-9
wvl_R = 638 * nm  # Red
wvl_G = 520 * nm  # Green
wvl_B = 450 * nm  # Blue

# resolution setting
h = 2160 #3840
w = 3840 #2160

pp = 3.6 * um  # pixel to pixel parameter

field_length = 1  # reconstruct distance

# scale factor
xy_scale = 0.03 
z_scale = 0.25

# read PLY file and convert to numpy array
with open('PointCloud_Dice_RGB.ply', 'rb') as f:
    plydata = plyfile.PlyData.read(f)
data = np.array(plydata.elements[1].data)

def k0(wvl):
    return (np.pi * 2) / wvl  # wave number

def h_RS(x1, y1, x2, y2, z, wvl):
    """
    RS impulse response
    """
    r = np.sqrt((x1-x2)**2 + (y1-y2)**2 + z**2)
    h = (z / (1j * wvl)) * (np.exp(1j*k0(wvl)*r) / r **2)
    return h

def anti(wvl, z, p):
    txy = wvl / (2 * p)
    t = (txy / np.sqrt(1 - txy ** 2)) * z
    return np.abs(t)

def RS_hologen(z, wvl):
    """
    hologram generation function
    """
    if wvl == wvl_R:
        color = 'red'
    elif wvl == wvl_G:
        color = 'green'
    else:
        color = 'blue'
    RS = np.zeros((h,w), dtype='complex64')  # hologram plane - real / imaginary part 생성
    for n in range(len(data)):
        x1 = data['x'][n] * xy_scale
        y1 = data['y'][n] * xy_scale
        z1 = data['z'][n] * z_scale
        a = anti(wvl, z1 , pp)   # anti aliasing factor
        RS_cash = np.zeros((h,w), dtype='complex64')
        for i in range(h):
            y2 = (i - h//2) * pp
            if y1 - a < y2 < y1 + a:
                for j in range(w):
                    x2 = (j - w//2) * pp  # pixel location
                    if x1 -a < x2 < x1 + a:  # anti alliasing 조건
                        RS_cash[i, j]= h_RS(x1, y1, x2 , y2 , z - z1, wvl_R) * data[color][n]
                else:
                    pass
            else:
                pass
        RS = RS + RS_cash
        print(n , " th cash done")
    return RS

def to_img(ch_Red, ch_Green, ch_Blue):
    r = np.imag(ch_Red)
    g = np.imag(ch_Green)
    b = np.imag(ch_Blue)
    img = np.zeros((h, w, 3))
    r = (r-np.min(r)) / np.max(r)  # normalize
    g = (g-np.min(g)) / np.max(g)
    b = (b-np.min(b)) / np.max(b)
    img[:, :, 0] = r
    img[:, :, 1] = g
    img[:, :, 2] = b
    plt.imshow(img)
    plt.imsave('RGB_img.bmp', img)
