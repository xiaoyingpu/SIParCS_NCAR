from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist


np.set_printoptions(precision=5, suppress=True)  # suppress scientific float notation


df = pd.read_csv("./csv/ssim_cmip5_cvdp_42.csv")

pd.set_option('display.max_rows', len(df))
print df

#plt.scatter(df.x, df.y)
#plt.show()

X = df.as_matrix()
lbl = X[:, 2]
X = X[:, :2]
Z = linkage(X, 'ward')

c, coph_dists = cophenet(Z, pdist(X))
print c

# calculate full dendrogram
plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    Z,
    labels = lbl,
    leaf_rotation=60.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()
