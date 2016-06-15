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
            print string_row
            l.append([float(i) for i in string_row])
    return np.array(l)




if len(sys.argv) != 2:
    print("Usage: frameworkpython dm_manifold.py <img dir>")

f_list = []


IS_TIF = False
if IS_TIF:
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
else:
    for f in os.listdir(sys.argv[1]):    # img dir as commandline arg
        if (f.endswith(".csv")):
            f_list.append(f)
    os.chdir(sys.argv[1])
    N = len(f_list)
    dm = np.zeros((N,N))

    for i_tuple in itertools.combinations(range(len(f_list)), 2):
        i, j = i_tuple
        print i , j
        i_dat = get_csv_array(f_list[i])
        j_dat = get_csv_array(f_list[j])

        s = 1-ssim(i_dat, j_dat)

        dm[i][j] = s
        dm[j][i] = s

# http://www.nervouscomputer.com/hfs/cmdscale-in-python/
# classical MDS
k = 2   # top 2 vectors for 2D vis
end = time.clock()
print("Building distance matrix: {}".format(end-start))

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


eigen = time.clock()
print("Finding eigenvectors: {}".format(eigen - end))


# Y: (n, p) array, col is a dimension

d = category.get_color_dic(os.path.abspath("."))
df = {}

for i in range(len(d)):
    df[i] = []

for i in range(N):
    index = d[category.model(f_list[i])]
    x = Y[:,0][i]
    y = Y[:,1][i]
    f = f_list[i]
    df[index].append([x,y,f])


for i in range(len(d)):
    df[i] = np.array(df[i])



fig, ax = plt.subplots()
# ax.scatter(Y[:,0], Y[:,1])
palette = np.array(sns.color_palette("hls", 25))

for i in range(len(d)):
    ax.scatter(df[i][:,0], df[i][:,1], \
            label = category.model(df[i][:,2][0]),\
            color = palette[i])

ann = []
for i in range(N):
    ann.append(ax.annotate(category.model(f_list[i]), xy = (list(Y[:,0])[i], list(Y[:,1])[i])))
mask = np.zeros(fig.canvas.get_width_height(), bool)

plt.tight_layout()
plt.legend()
# overlapping labels removal
fig.canvas.draw()
for a in ann:
    bbox = a.get_window_extent()
    x0 = int(bbox.x0)
    x1 = int(math.ceil(bbox.x1))
    y0 = int(bbox.y0)
    y1 = int(math.ceil(bbox.y1))

    s = np.s_[x0:x1+1, y0:y1+1]
    if np.any(mask[s]):
        a.set_visible(False)
        # a hack to display IPSL
        #if "IPSL" in a.get_text():
        #    a.set_visible(True)
    else:
        mask[s] = True
plt.show()



# csv persistance
DO_PERSISTENCE = False
if DO_PERSISTENCE:
    with open("out.csv", "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y","label"])
        for i in range(N):
            row = [Y[:,0][i], Y[:,1][i], category.model(f_list[i])]
            writer.writerow(row)

