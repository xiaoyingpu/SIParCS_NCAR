from __future__ import division
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from sklearn import manifold, datasets
from skimage.measure import compare_ssim  as ssim

from scipy.linalg import eigh as largest_eigh
from math import sqrt
import numpy as np
import itertools
import os, sys, cv2, csv


# replace the kNN, graph approach with a distance matrix,
# SSIM as the distance metric
# make sense????


if len(sys.argv) != 2:
    print("Usage: frameworkpython dm_manifold.py <img dir>")

f_list = []

for f in os.listdir(sys.argv[1]):    # img dir as commandline arg
    if (f.endswith(".tif") or f.endswith(".png")):
        f_list.append(f)
# change working directory
os.chdir(sys.argv[1])

N = len(f_list)
# distance matrix, n by n init to zeros
dm = np.zeros((N, N))

for i_tuple in itertools.combinations(range(len(f_list)), 2):
    i, j = i_tuple
    img1 = cv2.imread(f_list[i])
    img2 = cv2.imread(f_list[j])
    # to grey scale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    s = 1 - ssim(img1, img2)

    # symmetric matrix!
    # not sparse anymore!!!!
    dm[i][j] = s
    dm[j][i] = s
print dm
# http://www.nervouscomputer.com/hfs/cmdscale-in-python/
# classical MDS
k = 2   # top 2 vectors for 2D vis


# centering matrix
H = np.eye(N) - np.ones((N, N))/N
tau = -0.5 * H.dot(dm ** 2).dot(H)

# eigen- in descending order
evals, evecs = largest_eigh(tau)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:,idx]

w, = np.where(evals > 0)
L = np.diag(np.sqrt(evals[w]))
V = evecs[:,w]
Y = V.dot(L)

# Y: (n, p) array, col is a dimension


# csv persistance
with open("out.csv", "w+") as f:
    writer = csv.writer(f)
    for i in range(N):
        row = [Y[:,0][i], Y[:,1][i], f_list[i]]
        writer.writerow(row)

