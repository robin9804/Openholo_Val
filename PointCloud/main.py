import matplotlib.pyplot as plt
import time

import functions.Fresnel_fft
import functions.RS_integral
import functions.Fresnel_integral
import functions.ASM_fft


def main():
    method = 3
    # 1 : RS method, 2 : Fresnel integral, 3 : Angular Spectrum method, 4 : Fresnel FFT
    start = time.time()
    print("start")

    if method == 1:
        # R-S integral
        p = functions.RS_integral.RS('PointCloud_Dice_RGB.ply', angleY=22.5)  # initialize
        R1 = p.CalHolo()  # Generate hologram
        B1 = p.CalHolo('blue')
        G1 = p.CalHolo('green')
        # R1 = p.SSB(Red)           # encoding
        p.getRGBImage(R1, G1, B1, 'result/RS_DICE_1.bmp')       # get color image
        # p.getMonoImage(G1, 'result/3point')                   # get mono image
        del p, R1, B1, G1

    elif method == 2:
        # Fresnel Integral
        frsn = functions.Fresnel_integral.Frsn_Integral('PointCloud_Dice_RGB.ply', angleY=22.5)
        red2 = frsn.CalHolo()
        green2 = frsn.CalHolo('green')
        blue2 = frsn.CalHolo('blue')
        frsn.getRGBImage(red2, green2, blue2, 'result/FrsnI_DICE_1.bmp')
        del frsn, red2, green2, blue2

    elif method == 3:
        # ASM
        asm = functions.ASM_fft.ASM('PointCloud_Dice_RGB.ply', angleY=22.5)
        red3 = asm.CalHolo()
        green3 = asm.CalHolo('green')
        blue3 = asm.CalHolo('blue')
        asm.getRGBImage(red3, green3, blue3, 'result/ASM_DICE_1.bmp')
        del asm, red3, green3, blue3

    elif method == 4:
        # Fresnel FFT
        f = functions.Fresnel_fft.FrsnFFT('point_3.ply', angleY=22.5)
        red4 = f.CalHolo()
        green4 = f.CalHolo('green')
        blue4 = f.CalHolo('blue')
        f.getRGBImage(red4, green4, blue4, 'result/Frsn_FFT_DICE1.bmp')
        #f.getMonoImage(red, 'result/Frsn_point_1')
        del f, red4, green4, blue4

    else:
        print("Wrong Number")

    print("=====Total Time ", time.time() - start, " =====")


if __name__ == '__main__':
    main()



