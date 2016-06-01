#from pylab import scatter,text,show,cm,figure
#from pylab import subplot,imshow,NullLocator
#http://stackoverflow.com/questions/25516325/non-overlapping-scatter-plot-labels-using-matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn import manifold, datasets
import os, sys, cv2, re

if len(sys.argv) != 2:
    print("Usage: frameworkpython manifold.py <img dir>")
    exit()

f_list = []
label_list = []
X = []

for f in os.listdir(sys.argv[1]):    # img dir as commandline arg
    if (f.endswith(".tif") or f.endswith(".png")):
        f_list.append(f)
        #lbl = re.findall(r"#\d+", f)
        lbl = f[6:10]
        # label_list.append(lbl[0][1:])              # regex!
        label_list.append(f[1:7])
# change working directory
os.chdir(sys.argv[1])

for f in f_list:
    img = cv2.imread(f)
    X.append(list(img.flat))
# load the digits dataset
# 901 samples, about 180 samples per class
# the digits represented 0,1,2,3,4
#digits = datasets.load_digits(n_class=5)
#X = digits.data
#color = digits.target

# running Isomap
# 5 neighbours will be considered and reduction on a 2d space
Y = manifold.Isomap(5, 2).fit_transform(X)

# plotting the result
#figure(1)
fig, ax = plt.subplots()
ax.scatter(Y[:,0], Y[:,1])
#ann = []
#for i in range(len(label_list)):
#    ann.append(ax.annotate(label_list[i], xy = (list(Y[:,0])[i], list(Y[:,1])[i])))

#mask = np.zeros(fig.canvas.get_width_height(), bool)

#fig.canvas.show()

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
#    else:
#        mask[s] = True

plt.tight_layout()

plt.scatter(Y[:,0], Y[:,1], c='k', alpha=0.3, s=10)
for i in range(Y.shape[0]):
 plt.text(Y[i, 0], Y[i, 1], str(label_list[i]),
      fontdict={'weight': 'bold', 'size': 11})
plt.show()
