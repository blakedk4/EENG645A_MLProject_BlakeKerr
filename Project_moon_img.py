# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 08:24:07 2024

@author: swordsman
@edited by: Blake Kerr
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.stats import norm
from numpy import squeeze, ceil, real
from math import factorial
from scipy.special import gamma
from numpy.linalg import cholesky

def fourier_downsample(img, new_size):
    old_size = img.shape[0]
    assert img.shape[0] == img.shape[1], "Image must be square"

    # FFT
    F = fft2(img)
    F_shift = fftshift(F)

    # Compute crop indices
    center = old_size // 2
    half = new_size // 2

    F_crop = F_shift[
        center-half:center+half,
        center-half:center+half
    ]

    # Shift back
    F_crop = ifftshift(F_crop)

    # Inverse FFT
    img_small = np.real(ifft2(F_crop))

    # Energy scaling correction
    scale = (new_size / old_size) ** 2
    img_small *= scale

    return img_small.astype(np.float32)

##Aperture Function
def make_pupil(r1,r2,si):
    #r1=radius of moon in pixels
    #r2 coronagraph function or center support?
    #pixel l/w
    if(2*np.floor(si/2)==si):
        mi=int(np.floor(si/2));
        pupilx=np.zeros([si,si])
        for i in range(0,si-1):
            pupilx[i]=range(-mi,mi)
    if(2*np.floor(si/2)!=si):
         mi=int(np.floor(si/2));
         pupilx=np.zeros([si,si])
         for i in range(0,si-1):
             pupilx[i]=range(-mi,mi+1)
    pupily=np.transpose(pupilx)
    dist2=np.multiply(pupilx,pupilx)+np.multiply(pupily,pupily)
    dist=np.sqrt(dist2)
    pupil2=(dist<r1)
    pupil3=(dist>r2)
    pupil=np.multiply(pupil2.astype(int),pupil3.astype(int))
    #plt.figure()
    #plt.imshow(pupil)
    #plt.title("Aperture Function")
    return(pupil)

#Moon Simulation Function
def simulate_moon(D1,D2,F,lam,dt,background,si):
    #D1 is the actual aperture diameter in meters
    #D2 is the aperture size used to compute the sampling in the detector
    #F is the focal length of the telescope
    # lam is the wavelength of the light in units of meters
    # dt is the integration time in seconds
    # background is the black sky photon level

    path = os.getcwd() + '/EENG645A_MLProject_BlakeKerr/moon_img.txt'#combine directory with image name for full path
    opo9914d = np.genfromtxt(path, delimiter=",").astype('float')
    
    #Prealocate 1200 by 1200 image with 3 channels for RGB
    img = np.zeros((1200, 1200, 3))
    img[:, :1200, 0] = opo9914d[:, :1200]
    img[:, :1200, 1] = opo9914d[:, 1200:2400]
    img[:, :1200, 2] = opo9914d[:, 2400:3600]
    plt.figure()
    plt.imshow(img/255.0)#imshow wants RGB values from 0 to 1, so it is normalized
    plt.title("Source Image True")

    #Solve for pixel count and size for in focus moon
    dpix = lam * F / (2 * D2)
    dtheta = lam / (2 * D2)
    Moon_dist = 384400000.0                   # meters
    Moon_diameter = 3474800.0                 # meters
    Moon_pixels = Moon_diameter / (Moon_dist * dtheta)
    pixels = round(Moon_pixels)

    #Interpolate inside the fourier domain (Fourier Domain interpolator)
    f_interp_moon = np.zeros([pixels, pixels]).astype('complex')
    fmoon = fftshift(fft2(squeeze(img[:,:, 2])))#Uses Blue light since that should be the dominant color (Closest to gray) squeeze to cut out useless dimension
    half_pix = ceil(pixels / 2).astype('int')
    f_interp_moon[half_pix - 600: half_pix + 600,
                    half_pix - 600: half_pix + 600] = fmoon
    #print(pixels)
    moon_outline=make_pupil(pixels/2, 0, 6000)
    #print(pixels)
    moon = np.abs(ifft2(fftshift(f_interp_moon)))

    #Preallocate new array and multiply the resized source image to the aperture
    Source_img = np.zeros((6000,6000))
    Source_img[3001 - half_pix : 3001 + half_pix,
                3001 - half_pix : 3001 + half_pix] = moon #Ensure moon is in center of matrix
    Source_img=np.multiply(Source_img,moon_outline)
    #plt.figure()
    #plt.imshow(Source_img)
    #plt.title("Source Image Transformed")

    # Calculate energy from the moon and convert to desired unit of expected photons
    Intensity = 1000.0                          # w / m ^ 2 power per unit area hitting the moon (Solar power value)
    h = 6.62607015e-34                          # plancks constant
    c = 3.0e8                                   # speed of light in meters
    v = c / lam                                 # frequency of light
    moon_reflectivity = 0.10                    # moon's reflectivity is 10%
    photons = (D1*D1*Intensity * (Moon_dist * dtheta) *(Moon_dist * dtheta)* dt ) /(4*Moon_dist*Moon_dist* (h * v))#Radiometry formula lambersian reflection 
    #  Steradians from radiometry
    #  h*v is J/photon
    #  energy = (photons / (4.0 * np.pi * Moon_dist ** 2.0)) * np.pi * (D / 2.0)**2.0
    Sfac=np.sum(Source_img)/(np.sum(moon_outline)*moon_reflectivity)#Average value per pixel/reflectivity

    # Make Image reflect real energy values
    # moon_max = np.max(np.max(Source_img))
    norm_moon = np.divide(Source_img, Sfac)#Prob of photon from every point of the moon, moves reflectivity back to numerator
    photons_img = photons*np.multiply(norm_moon, moon_outline)+background #Photons in image is #from moon*prob of photon+background
    images = photons_img
    plt.figure()
    plt.imshow(images)
    plt.show()
    #moon_outline=make_pupil(3200//2, 0, 6000)
    moon_outline = make_pupil(3240//2*(si/6000),0,si)
    moon_outline[si//2,si//2]=1

    folder="Project_Data"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder,"aperture")
    np.save(filename, moon_outline)
    images=fourier_downsample(images, si)

    return images,moon_outline
'''
#ZernikeFunctions
#turb_freq=turbulence()

##Main
si=6000
D = 0.07                        # diameter of telescope in meters
Ds=0.1                          # diameter model for nyquist sampling with whole number pixels
obs = 0                         # background
lamb =575*10**(-9)              # wavelength of ligth in meters
f = 0.4                         # focal length in meters
dt=0.001     
#Nyquist=lamb*f/(2*.1)

photon_img,aperture = simulate_moon(D,Ds, f, lamb,dt,obs,si)

star_sz=1
photon_count=1*photon_img.max()
locx=1000
locy=3000
frame=np.zeros([si,si])
frame[locy:locy+star_sz,locx:locx+star_sz]=1
frame2=np.bitwise_or(frame.astype(np.uint8),aperture.astype(np.uint8))-aperture.astype(np.uint8)
star_img=frame2.astype(np.uint64)*photon_count
#plt.figure()
#plt.imshow(frame2)
#plt.title("Frame Test")

Full_img=photon_img+star_img
plt.figure()
plt.imshow(Full_img)
plt.title("Full Image")

#Diffracted Light is an atmospheric effect?
#Atmosphere Effect #var is (D/Ds)^2
turb_freq=turbulence(si)#Second Aperture?
turb_freq=turb_freq/np.max(turb_freq)
Full_img=Full_img/np.max(Full_img)
Img_freq=(fft2(Full_img))
Img=Img_freq*turb_freq
out_field=np.real(ifft2((Img)))
#plt.figure()
#plt.imshow(zern_phase)
#plt.title("Zern Phase Test")
#plt.figure()
#plt.imshow(np.abs(turb_freq))
#plt.plot(Img)
#plt.title("fourier space")
plt.figure()
plt.imshow(np.abs(out_field))
plt.title("Out Field Test")



plt.show()
'''