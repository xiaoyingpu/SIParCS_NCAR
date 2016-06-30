import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

# read scatter dataset; pandas?

with open("./csv/ssim_cmip5_with_label.csv", "r") as f:
    df = pd.read_csv(f)

X = np.array(df.ix[:,:"y"])

n_clusters = 10
k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
k_means.fit(X)
# labels
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels_unique = np.unique(k_means_labels)


# Plot result

fig,ax = plt.subplots(figsize=(20,10))
colors = sns.color_palette("hls", n_colors = n_clusters)

# KMeans
for k, col in zip(range(n_clusters), colors):
    my_members = k_means_labels == k
    cluster_center = k_means_cluster_centers[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            linestyle = " ", markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=10)
ax.set_title('KMeans w/ n_clusters = {}'.format(n_clusters))
#ax.set_xticks(())
#ax.set_yticks(())


plt.show()
