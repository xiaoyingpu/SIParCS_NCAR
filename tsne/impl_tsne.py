# That's an impressive list of imports.
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel
# Random state.
RS = 20150101

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib

# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

import os, sys, cv2
import itertools
from skimage.measure import compare_ssim  as ssim


def get_model(s):
    return s.split("_", 1)[0].split("-",1)[0]

def color_index(f_list):
    y = []
    d = {}
    i = 0
    for f in f_list:
        m = get_model(f)
        if m not in d:
            d[m] = i
            i += 1
        y.append(d[m])
    return y, d


def scatter(x):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", 25))

    # We create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    for i in range(len(x)):
        ax.scatter(x[i][:,0], x[i][:,1], lw=0, s=40,
                    c = palette[i],
                    label = x[i][:,2][0])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    plt.legend()
    ax.axis('off')
    ax.axis('tight')


    # We add the labels for each digit.
    txts = []
    #for i in range(10):
    #    # Position of each label.
    #    xtext, ytext = np.median(x[colors == i, :], axis=0)
    #    txt = ax.text(xtext, ytext, str(i), fontsize=24)
    #    txt.set_path_effects([
    #        PathEffects.Stroke(linewidth=5, foreground="w"),
    #        PathEffects.Normal()])
    #    txts.append(txt)

    #return f, ax, sc, txts


if len(sys.argv) != 2:
    print("Usage: frameworkpython impl_tsne.py <img dir>")
    exit()

f_list = []
label_list = []
X = []

for f in os.listdir(sys.argv[1]):    # img dir as commandline arg
    if (f.endswith(".tif") or f.endswith(".png")):
        f_list.append(f)
        label_list.append(get_model(f))
os.chdir(sys.argv[1])



N = len(f_list)
X = np.zeros((N,N))

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
    X[i][j] = s
    X[j][i] = s
#for f in f_list:
#    img = cv2.imread(f)
#    X.append(list(img.flat))

# color dictionary: model name -> color code
y, d = color_index(f_list)
y = np.hstack(y)
# We first reorder the data points according to the handwritten numbers.
#X = np.vstack([digits.data[digits.target==i]
#                   for i in range(10)])
#y = np.hstack([digits.target[digits.target==i]
#                   for i in range(10)])

digits_proj = TSNE(random_state=RS, metric="precomputed").fit_transform(X)


# build data frame:
df = {}
for i in range(len(d)):
    df[i] = []

for i in range(N):
    m = get_model(f_list[i])
    index = d[m]
    x = digits_proj[:,0][i]
    y = digits_proj[:,1][i]
    df[index].append([x,y,m])

for i in range(len(d)):
    df[i] = np.array(df[i])

print(df[0])
scatter(df)
plt.show()

