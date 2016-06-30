from __future__ import division
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.pylab import cm

from sklearn import manifold, datasets
from skimage.measure import compare_ssim  as ssim

import pandas as pd

from scipy.linalg import eigh as largest_eigh
import math
from math import sqrt
import numpy as np
import itertools
import os, sys, cv2, csv

import category
# replace the kNN, graph approach with a distance matrix,
# SSIM as the distance metric
# make sense????

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

import time
start = time.clock()


def get_csv_array(fname):
    l = []
    with open(fname) as f:
        reader = csv.reader(f)
        for row in reader:
            string_row = row[0].split()
            l.append([float(i) for i in string_row])
    return np.array(l)





def get_distance_matrix(img_dir, f_list):
    #f_list, IS_IMG = get_f_list(img_dir)

    IS_IMG = False
    os.chdir(img_dir)
    N = len(f_list)
    dm = np.zeros((N, N))
    if IS_IMG:
        # distance matrix, n by n init to zeros
        for i_tuple in itertools.combinations(range(len(f_list)), 2):
            i, j = i_tuple
            img1 = cv2.imread(f_list[i])
            img2 = cv2.imread(f_list[j])
            # to grey scale
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            s = 1 - ssim(img1, img2)    # so that distance makes sense

            # symmetric matrix!
            # not sparse anymore!!!!
            dm[i][j] = s
            dm[j][i] = s
    else:   # csv's, right
        print("reading csv files as images")
        f_list = f_list.as_matrix()
        for i_tuple in itertools.combinations(range(len(f_list)), 2):
            i, j = i_tuple
            i_dat = get_csv_array(f_list[i][0].lstrip() + ".csv")
            j_dat = get_csv_array(f_list[j][0].lstrip() + ".csv")

            s = 1-ssim(i_dat, j_dat)

            dm[i][j] = s
            dm[j][i] = s
    return dm

# -----------main-----------

if len(sys.argv) != 2:
    print("Usage: frameworkpython dm_manifold.py <img dir>")

script_path = os.path.abspath(".")


# CVDP provides 42 models with mean scores and RMS diff
# in models.csv file in the same directory

with open("models.csv") as f:
    pd_df = pd.read_csv(f, delimiter=",")

    f_list = pd_df.ix[:,1:2]

N = len(f_list)

# note: the working dir will be changed if the first
# branch is taken
if not os.path.isfile("dm.txt"):
    # need to compute from scratch
    print("Generating distance matrix")
    dm = get_distance_matrix(sys.argv[1], f_list)
    # since computing distance matrix is expensive
    # save a copy for later use
    with open ("dm.txt", "w") as f:
        np.savetxt(f, dm)
else:
    # read the distance matrix
    print("Loading distance matrix from file")
    dm = np.loadtxt("dm.txt")

# http://www.nervouscomputer.com/hfs/cmdscale-in-python/
# classical MDS
# Classical MDS assumes Euclidean distances. So this is not applicable for direct dissimilarity ratings. -wiki



# Y: (n, p) array, col is a dimension
Y = manifold.MDS(n_components=2, dissimilarity='precomputed').fit_transform(dm)
Y = np.array(Y)




# ---------- persistence ------
f_persist = "ssim_cmip5_cvdp.csv"

model_list = pd_df["short_name"].tolist()
pd_df.columns = pd_df.columns.map(str.strip)
print pd_df.columns.values.tolist()
f_list = pd_df["filename"].tolist()

# csv persistance
DO_PERSISTENCE = True
if DO_PERSISTENCE:
    os.chdir(os.path.join(script_path, "csv"))
    with open(f_persist, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y","label", "fname"])
        for i in range(N):
            x = Y[:,0][i]
            y = Y[:,1][i]
            row = [x, y, model_list[i], f_list[i]]
            writer.writerow(row)

