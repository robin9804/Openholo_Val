clc;
fclose('all');

% unit
mm = 1e-3;
um = mm*mm;
nm = um*mm;

% inline command
asm_kernel = @(f, wvl, x, y, res, pp) exp(-2j*pi*f .* sqrt( wvl^-2 - ((x/pp - 0.5)./pp/res).^2 - ((y/pp - 0.5)./pp/res).^2));     % angular spectrum method kernel
norm2int = @(img) uint8(255.*img./max(max(img)));

% parameters
h = 1080; v = 1920;
wvl = 520 * nm;
pp = 3.45*um; % pixel pitch
res = v/2;
ReconstLength = 0.8;

% input file name
fim_py = 'phase_3point_3.bmp';
fre_py = 'real_3point_3.bmp';

% Convert to complex image
im_py = imread(fim_py);
im_py = im_py(:, 421:1500, 1);
re_py = imread(fre_py);
re_py = re_py(:,421:1500,1);
im_py = double(im_py);
re_py = double(re_py);

% adapt ASM
r = (-pp*h/2 + pp/2):pp:(pp*h/2 - pp/2);
c =  r; 
[C, R] = meshgrid(c, r);

% Case: phase and amplitude encoded image
im = im_py;
re = re_py;
re = (re/256) * pi * 2;

imag = im .* sin(re);
real = im .* cos(re); 

% Complex image
%ch = real + 1j*imag;
ch = re_py + 1j*im_py;
A = fftshift(fft2(fftshift(ch)));

%%
% Reconstruction part

figure;
for zz = (300:10:600)*mm
%for zz = ReconstLength * mm
    p = asm_kernel(zz, wvl, C, R, res, pp);
    Az1 = A .* p;
    EI = fftshift(ifft2(fftshift(Az1)));
    I_rec = EI .* conj(EI);
    I_rec = I_rec / max(max(I_rec)); %normalize term
    I_rec = 255 .* I_rec;
    I_rec = uint8(I_rec);
    %imwrite(I_rec, sprintf('OPH_point_3_%.2fmm.jpg', zz/mm+20));
    imagesc(I_rec);
    title(sprintf('%.2f mm', zz/mm));
    pause(0.2);
end
