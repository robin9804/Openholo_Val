import matplotlib.pyplot as plt
import time

from RS_integral import *
import Fresnel_integral
import encoding
import ASM_fft
import Fresnel_fft

def main():
    start = time.time()
    print("start")
    
    # R-S Integral
    '''
    p = RS('point_3.ply')       # initialize
    Red = p.CalHolo(p.wvl_R)    # Generate hologram
    Red = p.SSB(Red)           # encoding
    blue = p.CalHolo(p.wvl_B)
    Green = p.CalHolo(p.wvl_G)
    p.getRGBImage(Red, Green, blue, 'result/point3.bmp')    # get color image
    p.getMonoImage(Red, 'result/3point')                   # get mono image
    '''
    
    # Fresnel Integral
    '''
    frsn = Fresnel_integral.Frsn_Integral('point_3.ply', angleY=10)
    red = frsn.CalHolo(frsn.wvl_R)
    green = frsn.CalHolo(frsn.wvl_G)
    blue = frsn.CalHolo(frsn.wvl_B)
    frsn.getRGBImage(red, green, blue, 'result/Frsn_Integral_point3.bmp')
    '''

    # ASM
    '''
    asm = ASM_fft.ASM('point_3.ply')
    red = asm.CalHolo(asm.wvl_R)
    #red = asm.SSB(red)
    asm.getMonoImage(red, 'result/ASM_point_1')
    '''
    # Fresnel FFT
    f = Fresnel_fft.Frsn_FFT('point_3.ply', angleY=10)
    red = f.CalHolo(f.wvl_R)
    g = f.CalHolo(f.wvl_G)
    b = f.CalHolo(f.wvl_B)
    #f.getMonoImage(red, 'result/Frsn_point_1')
    f.getRGBImage(red, g, b, 'result/Frsn_point_1.bmp')

    # check
    print(time.time() - start, ' is time')
    plt.imshow(np.imag(red))
    plt.show()



if __name__ == '__main__':

    main()



