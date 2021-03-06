#from pylab import scatter,text,show,cm,figure
#from pylab import subplot,imshow,NullLocator
#http://stackoverflow.com/questions/25516325/non-overlapping-scatter-plot-labels-using-matplotlib
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
from matplotlib.pylab import cm
import numpy as np
import math
from sklearn import manifold, datasets
import os, sys, cv2, re

import csv
import category

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
        label_list.append(category.model(f))
# change working directory
os.chdir(sys.argv[1])



N = len(f_list)

for f in f_list:
    img = cv2.imread(f)
    X.append(list(img.flat))

# running Isomap
# 5 neighbours will be considered and reduction on a 2d space
Y = manifold.Isomap(5, 2).fit_transform(X)

# persist csv
PERSISTENCE = True

if PERSISTENCE:
    with open("out.csv", "w+") as f:
        writer = csv.writer(f)
        for i in range(len(Y[:,0])):
            row = [Y[:,0][i], Y[:,1][i],f_list[i], label_list[i]]   # filename, model name
            writer.writerow(row)

# categorize datapoints by model
color_dic = category.get_color_dic(os.path.abspath("."))

df = {}
for i in range(len(color_dic)):
    df[i] = []

for i in range(N):
    index = color_dic[label_list[i]]
    x = Y[:,0][i]
    y = Y[:,1][i]
    f = f_list[i]
    df[index].append([x,y,f])

# convert to np array
for i in range(len(color_dic)):
    df[i] = np.array(df[i])



# plotting the result
#figure(1)
fig, ax = plt.subplots()
# ax.scatter(Y[:,0], Y[:,1])

for i in range(len(color_dic)):
    ax.scatter(df[i][:,0], df[i][:,1], \
            label = category.model(df[i][:,2][0]),\
            color = cm.Dark2(i/float(len(color_dic))))

ann = []
for i in range(len(label_list)):
    ann.append(ax.annotate(label_list[i], xy = (list(Y[:,0])[i], list(Y[:,1])[i])))
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
        if "IPSL" in a.get_text():
            a.set_visible(True)
    else:
        mask[s] = True
plt.show()

#plt.scatter(Y[:,0], Y[:,1], c='k', alpha=0.3, s=10)
#for i in range(Y.shape[0]):
# plt.text(Y[i, 0], Y[i, 1], str(label_list[i]),
#      fontdict={'weight': 'bold', 'size': 11})
