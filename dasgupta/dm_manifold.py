import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from sklearn import manifold, datasets
from skimage.measure import compare_ssim  as ssim

from scipy.linalg import eigh as largest_eigh
from math import sqrt
from __future__ import division
import numpy as np
import itertools
import os, sys, cv2


# replace the kNN, graph approach with a distance matrix,
# SSIM as the distance metric
# make sense????

def get_xy(i, eigval, eigvec):
    """
    the i-th data point
    """
    # 2nd eigenvalue, index = 1
    x = sqrt(eigval[1]) * eigvec[:,1][i]
    # 1st eigenvalue
    y = sqrt(eigval[0]) * eigvec[:,0][i]
    print x, y
    return x, y





if len(sys.argv) != 2:
    print("Usage: frameworkpython dm_manifold.py <img dir>")

f_list = []

for f in os.listdir(sys.argv[1]):    # img dir as commandline arg
    if (f.endswith(".tif") or f.endswith(".png")):
        f_list.append(f)
# change working directory
os.chdir(sys.argv[1])

# distance matrix, n by n init to zeros
dm = np.zeros((len(f_list), len(f_list)))

for i_tuple in itertools.combinations(range(len(f_list)), 2):
    i, j = i_tuple
    img1 = cv2.imread(f_list[i])
    img2 = cv2.imread(f_list[j])
    # to grey scale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    s = ssim(img1, img2)

    # symmetric matrix!
    # not sparse anymore!!!!
    dm[i][j] = s
    dm[j][i] = s


# classical MDS
N = len(f_list)
k = 2   # top 2 vectors for 2D vis


# get centering matrix




# eigen- in ascending order
#eigenvals, eigenvecs = largest_eigh(dm, eigvals = (N - k, N - 1))

for i in range(N):
    get_xy(i, eigenvals, eigenvecs)

# csv persistance

# visualization
