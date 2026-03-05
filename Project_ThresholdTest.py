"""
@author: Blake Kerr
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.fft import fft2, ifft2, fftshift
from scipy.stats import norm
from scipy import ndimage
from numpy import squeeze, ceil, real
from math import factorial
from scipy.special import gamma
from numpy.linalg import cholesky
from PIL import Image, ImageSequence
from sklearn.metrics import roc_curve, auc

def threshtest(threshold,image,aperture):
    moon_mask=(np.ones(np.shape(aperture))-aperture)*image
    bitmask = (moon_mask > threshold).astype(int)
    structure=np.ones((2,2))
    expanded=ndimage.binary_dilation(bitmask,structure=structure)
    labels, n = ndimage.label(expanded)
    if n == 0: 
        centers = None
    else:
        # compute area (size) for each label
        sizes = ndimage.sum(expanded, labels, range(1, n+1))
        # find the index of the largest region
        largest_index = np.argmax(sizes) + 1
        # compute centroid for only the largest region
        centers = ndimage.center_of_mass(expanded, labels, largest_index)
        #centers = ndimage.center_of_mass(expanded, labels, range(1, n+1))
    if not centers:
        y,x=-1,-1
    else:
        y, x = centers
    return int(np.round(x)),int(np.round(y))

def err_calc(strn,x,y):
    base = os.path.basename(strn)
    base = os.path.splitext(base)[0]

    # Split into parts
    parts = base.split("_")
    x_act=int(parts[1])
    y_act=int(parts[3])
    intensity=int(parts[5])
    if x==-1:
        err = 1000
    else:
        err=np.sqrt((x_act-x)**2+(y_act-y)**2)
    '''
    print("x is:",x,"   actual x is:",x_act)
    print("y is:",y,"   actual y is:",y_act)
    print("The intensity was:",intensity,"   Fraction of Max Lunar Intensity:",intensity/6101)
    print("Error is:",err)
    '''
    return err,intensity/6101

threshold = 250
strn="/remote_home/aperture.npy"
strn2="Project_Data/x_100_y_100_int_61004.npy"
aperture = np.load(strn)

folder = "/remote_home/Project_Data"
os.makedirs(folder, exist_ok=True)
npy_files = glob.glob(os.path.join(folder, "*.npy"))
npy_files = npy_files[1:]
print(npy_files)
err=np.zeros(np.size(npy_files))
intensity=np.zeros(np.size(npy_files))
i=0
scores = []  # confidence values
labels = []  # 0 or 1 if star present
bound = 10   # pixel distance threshold to count as "found"
for file_path in npy_files:
    image=np.load(file_path)
    #image[100,100]=200
    x,y=threshtest(threshold,image,aperture)
    #print(x,y)
    print("\nSample:",i+1)
    err[i],intensity[i] = err_calc(file_path,x,y)
    #confidence = np.exp(-err[i] / 10)
    if (x, y) != (-1, -1):
    #Use detected pixel value as confidence
        confidence = image[y, x] / 6101.0  # normalize intensity to [0,1]
    else:
        confidence = 0.0 
    labels.append(1 if err[i] < bound else 0)  # 1 = correctly found
    scores.append(confidence)  # use intensity as confidence
    i=i+1
print("\nThe Total Error is:",np.round(np.sum(err)))
print("The Mean Error is:",np.round(np.mean(err)))

correct = np.sum(err < 10)
total = len(err)
accuracy = correct / total
print("The Accuracy is:", accuracy)

plt.figure()
#plt.hist(err,intensity)
# Split data
values_error0 = intensity[err < bound]
values_error1 = intensity[err > bound]
bins = np.linspace(0, 1, 21)
# Plot
plt.hist(values_error0, bins=bins, alpha=0.6, label='Star Found', color='blue')
plt.hist(values_error1, bins=bins, alpha=0.6, label='Star Lost', color='red')
plt.xlabel("Intensity of Star Realative to Moon")
plt.ylabel("Quantity of Stars")
plt.title("Threshold Test Results")
plt.legend()
plt.xticks(np.arange(0, 1, 0.1))
plt.show()
#Reciever Operator Characteristic
# plot_roc_comparison.py
# Save to disk
folder = "./figures"
os.makedirs(folder, exist_ok=True)
np.save(os.path.join(folder, "labels.npy"), labels)
np.save(os.path.join(folder, "scores.npy"), scores)
np.save(os.path.join(folder, "accuracy.npy"),accuracy)

print("Threshold evaluation results saved!")
'''
aperture = np.load(strn)
aperture = make_pupil(3144//2,0,6000)
aperture[6000//2,6000//2]=1

folder="Project_Data"
os.makedirs(folder, exist_ok=True)
filename = os.path.join(folder,"aperture")
np.save(filename, aperture)
'''