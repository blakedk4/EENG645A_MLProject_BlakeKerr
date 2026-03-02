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
from scipy import ndimage

strn="Project_Data/x_990_y_626_int_42278.npy"
strn="Project_Data/x_616_y_248_int_531.npy" #Near Impossible
strn="Project_Data/x_40_y_66_int_5020.npy" #Obv
strn="Project_Data/x_685_y_745_int_1210.npy" #Obv
#strn="Project_Data/aperture.npy"
image_new = np.load(strn)
plt.figure()
plt.imshow(image_new)
plt.title(strn)

strn="Project_Data/aperture.npy"
aperture = np.load(strn)
image=image_new
threshold=250

moon_mask=(np.ones(np.shape(aperture))-aperture)*image
bitmask = (moon_mask > threshold).astype(int)
structure=np.ones((2,2))
expanded=ndimage.binary_dilation(bitmask,structure=structure)
labels, n = ndimage.label(expanded)
plt.figure()
plt.imshow(moon_mask)
plt.title('mask')
plt.show()
'''
#aperture = np.load(strn)
aperture = make_pupil(3144//2//2,0,si)
aperture[si//2,si//2]=1

folder="Project_Data"
os.makedirs(folder, exist_ok=True)
filename = os.path.join(folder,"aperture")
np.save(filename, aperture)
'''