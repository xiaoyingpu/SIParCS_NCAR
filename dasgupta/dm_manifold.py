from __future__ import division
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.pylab import cm

from sklearn import manifold, datasets
from skimage.measure import compare_ssim  as ssim


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


def get_f_list(img_dir):
    f_list = []
    IS_IMG = False
    for f in os.listdir(img_dir):
        if f.endswith(".tif") or f.endswith(".png"):
            f_list.append(f)
            IS_IMG = True
        elif f.endswith(".csv"):
            f_list.append(f)
    return f_list, IS_IMG



def get_distance_matrix(img_dir):
    f_list, IS_IMG = get_f_list(img_dir)

    os.chdir(img_dir)
    N = len(f_list)
    dm = np.ones((N, N))
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
    else:
        for i_tuple in itertools.combinations(range(len(f_list)), 2):
            i, j = i_tuple
            i_dat = get_csv_array(f_list[i])
            j_dat = get_csv_array(f_list[j])

            s = 1-ssim(i_dat, j_dat)

            dm[i][j] = s
            dm[j][i] = s
    return dm

# -----------main-----------

if len(sys.argv) != 3:
    print("Usage: frameworkpython dm_manifold.py <img dir> <sst/psl>")
    exit()

script_path = os.path.abspath(".")

variable_name = sys.argv[2]


f_list, not_used= get_f_list(sys.argv[1])

N = len(f_list)

# note: the working dir will be changed if the first
# branch is taken
dm_fname = "dm-" + variable_name + "dm.txt"
# delete these -------
#dm_fname = "dm_timecurve.txt"
#variable_name = "timecurve"
# delete these -----

if not os.path.isfile(dm_fname):
    # need to compute from scratch
    print("Generating distance matrix")
    dm = get_distance_matrix(sys.argv[1])
    # since computing distance matrix is expensive
    # save a copy for later use
    with open (dm_fname, "w") as f:
        np.savetxt(f, dm)
else:
    # read the distance matrix
    print("Loading distance matrix from file")
    dm = np.loadtxt(dm_fname)

# http://www.nervouscomputer.com/hfs/cmdscale-in-python/
# classical MDS
# Classical MDS assumes Euclidean distances. So this is not applicable for direct dissimilarity ratings. -wiki

#k = 2   # top 2 vectors for 2D vis
#end = time.clock()
#print("Building distance matrix: {}".format(end-start))
#
## centering matrix
#H = np.eye(N) - np.ones((N, N))/N
#tau = -0.5 * H.dot(dm ** 2).dot(H)
#
## eigen- in descending order
#evals, evecs = largest_eigh(tau)
#idx = np.argsort(evals)[::-1]
#evals = evals[idx]
#evecs = evecs[:,idx]
#
#w, = np.where(evals > 0)
#L = np.diag(np.sqrt(evals[w]))
#V = evecs[:,w]
#Y = V.dot(L)
#
#
#eigen = time.clock()
#print("Finding eigenvectors: {}".format(eigen - end))


# Y: (n, p) array, col is a dimension
Y = manifold.MDS(n_components=2, dissimilarity='precomputed').fit_transform(dm)
Y = np.array(Y)

d = category.get_color_dic(os.path.abspath(sys.argv[1]))
df = {}
for i in range(len(d)):
    df[i] = []
print N
for i in range(N):
    index = d[category.model(f_list[i])]
    x = Y[:,0][i]
    y = Y[:,1][i]
    f = f_list[i]
    df[index].append([x,y,f])
    print [x, y, f]


for i in range(len(d)):
    df[i] = np.array(df[i])

# ---------- plotting ------------
#f_persist = "csv/ssim_cmip5_with_label_{}.csv".format(variable_name)
#csv_path = os.path.join(script_path, f_persist)
#print csv_path
#if os.path.isfile(csv_path):
#    print "csv from last iter found, but not reading"
#
#
#fig, ax = plt.subplots()
## ax.scatter(Y[:,0], Y[:,1])
#palette = np.array(sns.color_palette("hls", len(d)))
#for i in range(len(d)):
#    lbl = category.model(df[i][:,2][0])
#    ax.scatter(df[i][:,0], df[i][:,1], \
#            label = lbl,\
#            color = palette[i])
#    print i, palette[i], lbl
#
##ax.scatter(-0.038728473132,0.0194969157674, label = "x", color=palette[8])
##print df[8][:,0],df[8][:,1]
#ann = []
#for i in range(N):
#    ann.append(ax.annotate(category.model(f_list[i]), xy = (list(Y[:,0])[i], list(Y[:,1])[i])))
#mask = np.zeros(fig.canvas.get_width_height(), bool)
#
#plt.tight_layout()
#plt.legend()
## overlapping labels removal
#fig.canvas.draw()
#for a in ann:
#    bbox = a.get_window_extent()
#    x0 = int(bbox.x0)
#    x1 = int(math.ceil(bbox.x1))
#    y0 = int(bbox.y0)
#    y1 = int(math.ceil(bbox.y1))
#
#    s = np.s_[x0:x1+1, y0:y1+1]
#    if np.any(mask[s]):
#        a.set_visible(False)
#        # a hack to display IPSL
#        #if "IPSL" in a.get_text():
#        #    a.set_visible(True)
#    else:
#        mask[s] = True
#plt.show()



# csv persistance
DO_PERSISTENCE = True
if DO_PERSISTENCE:
    os.chdir(os.path.join(script_path, "csv"))
    with open("ssim_cmip5_with_label_{}.csv".format(variable_name), "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y","label", "fname"])
        for i in range(N):
            row = [Y[:,0][i], Y[:,1][i], category.model(f_list[i]), f_list[i]]
            writer.writerow(row)

