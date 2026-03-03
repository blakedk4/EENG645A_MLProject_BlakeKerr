"""
@author: Blake Kerr
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.fft import fft2, ifft2, fftshift
from scipy.stats import norm
from numpy import squeeze, ceil, real
from math import factorial
from scipy.special import gamma
from numpy.linalg import cholesky
from PIL import Image, ImageSequence

from Project_moon_img import simulate_moon
from Project_Make_Phase_Screen_Short import full_otf
from Project_Make_Phase_Screen_Short import make_pupil
import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift



def star_addition(locx,locy,photon_count,Img_freq,otf,background,frame):
    star_sz=1
    frame[locy:locy+star_sz,locx:locx+star_sz]=1
    frame2=np.bitwise_or(frame.astype(np.uint8),aperture.astype(np.uint8))-aperture.astype(np.uint8)
    star_img=frame2.astype(np.uint16)*photon_count
    Img_freq_star=Img_freq+fft2(star_img)
    
    Img=Img_freq_star*otf
    out_field=np.abs(np.real(ifft2((Img))))
    lam = out_field + background           # ensure background >=0
    out_data = np.random.poisson(lam)

    filename = f"x_{x}_y_{y}_int_{photon_count:.0f}.npy"
    return out_data,filename


##Main
si=1024
D = 0.07                        # diameter of telescope in meters
Ds=0.1                          # diameter model for nyquist sampling with whole number pixels
obs = 1                         # background #dt*
lamb =575*10**(-9)              # wavelength of ligth in meters
f = 0.4                         # focal length in meters
dt=0.001     
#Nyquist=lamb*f/(2*.1)
photon_img,aperture = simulate_moon(D,Ds, f, lamb,dt,obs,si)
photon_img=photon_img.astype(np.float32)
print("Moon Generated")
otf=full_otf(si)
Img_freq=(fft2(photon_img))
print("OTF Generated")

#strn= os.getcwd() + "/EENG645A_MLProject_BlakeKerr/moonjune12gain5000pt5ms.tif"#
#image_new = Image.open(strn)
#data=np.array(image_new)
B_est=58#np.round(np.median(data[0,:]))

valid_positions = np.argwhere(aperture == 0)
frame=np.zeros([si,si])

#Star and Atmospheric Turbulence Addition/Randomization
data_samples=10
all_out_data=[]
all_filenames=[]

#Randomize variables and put in for loop. Check to ensure no starts infront of moon
for ii in range(0,data_samples):
    idx = np.random.randint(len(valid_positions))
    y, x = valid_positions[idx]
    count=np.random.uniform(0,photon_img.max())#10*photon_img.max()#
    out_data,filename=star_addition(x,y,count,Img_freq.copy(),otf,B_est,frame.copy())
    
    all_out_data.append(out_data)
    all_filenames.append(filename)
    print("completed star:",ii)

print("Saving")
#plt.show()
folder = "/remote_home/Project_Data"
os.makedirs(folder, exist_ok=True)

for img, fname in zip(all_out_data, all_filenames):
    filename = os.path.join(folder,fname)
    np.save(filename, img)

print("Done")