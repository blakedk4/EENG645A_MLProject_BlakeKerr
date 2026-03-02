import numpy as np
from PIL import Image, ImageSequence
from numpy import squeeze, ceil, real
from scipy.fft import fft2, ifft2, fftshift, fft, ifft
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from scipy.stats import t
def make_pupil(r1,r2,si):
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
        return(pupil)

def make_otf(scale,cpupil): #Telescope otf    
    psf=fft2(cpupil)
    psf=abs(psf)
    psf=np.multiply(psf, psf)
    spsf=np.sum(psf)
    norm_psf=scale*psf/spsf;
    otf=fft2(norm_psf)        
    return(otf,norm_psf)

def make_short_otf2(r1,dx,si,ro,alpha): #Atmospheric OTF
    import numpy as np
    from scipy.fft import fftshift
    #r1 is diameter
    pupilx=np.zeros((si,si));
    otf=np.zeros((si,si));
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
    temp=np.power((dx*dist/ro),(5/3))
    #temp3=(np.ones((si,si))-dist/(2*r1/dx)+.0000001)
    temp3=dx*dist/r1
    
    temp4=np.power(temp3,(1/3))
    temp5=np.ones((si,si))-alpha*temp4
    binmap=(temp5>0)
    temp6=np.multiply(temp5,binmap)
    temp2=-3.44*np.multiply(temp,temp6)
    otf=np.exp(temp2)
    otf2=fftshift(np.multiply(otf,binmap))
    return(otf2)
def full_otf(si):
    r1=3144/2
    cpupil = make_pupil(r1,0,si)
    [opt_otf,psf]=make_otf(1,cpupil)
    short_otf=make_short_otf2(.035,.14/si,si,.016,1)
    #r1=telescope diameter/aperture diameter
    #dx=sampling interval
    #si=output size
    #ro=fried parameter
    #alpha=weight
    otf=opt_otf*short_otf
    norm_otf=otf/np.max(otf)
    return(norm_otf)