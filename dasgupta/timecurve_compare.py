from __future__ import division

from sklearn import manifold, datasets
from skimage.measure import compare_ssim  as ssim


from scipy.linalg import eigh as largest_eigh
import math
from math import sqrt
import numpy as np
import itertools
import os, sys, cv2, csv

# replace the kNN, graph approach with a distance matrix,
# SSIM as the distance metric
# make sense????


import pandas as pd


# -----------main-----------


script_path = os.path.abspath(".")


# CVDP provides 42 models with mean scores and RMS diff
# in models.csv file in the same directory

with open("time.txt") as f:
    f_list = np.loadtxt(f, dtype = str)
    f_list = f_list[:,0]

print f_list


N = len(f_list)

# note: the working dir will be changed if the first
# branch is taken
f_dm = "dm_timecurve.txt"
if not os.path.isfile(f_dm):
    print("oops")
else:
    # read the distance matrix
    print("Loading distance matrix from file")
    dm = np.loadtxt(f_dm)

# Y: (n, p) array, col is a dimension
Y = manifold.MDS(n_components=2, dissimilarity='precomputed').fit_transform(dm)
Y = np.array(Y)




# ---------- persistence ------
f_persist = "ssim_walsh_timecurve.csv"


# csv persistance
DO_PERSISTENCE = True
if DO_PERSISTENCE:
    os.chdir(os.path.join(script_path, "csv"))
    with open(f_persist, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y","timelabel"])
        for i in range(N):
            x = Y[:,0][i]
            y = Y[:,1][i]
            row = [x, y, f_list[i]]
            writer.writerow(row)

