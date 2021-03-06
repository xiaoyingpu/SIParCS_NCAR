import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
from matplotlib.font_manager import FontProperties


df = pd.read_csv("./csv/ssim_cmip5_cvdp_42.csv")
grp = df.groupby("label")

sns.set_style("white")
sns.set_palette("hls", n_colors=len(grp))

fig, ax = plt.subplots(figsize=(8, 6), dpi = 96) # in inches
#palette = np.array(sns.color_palette("hls", 25))

for name, group in grp:
    plt.plot(group.x, group.y, marker = ".",ms = 10, linestyle = " ", label = name)
# was 10 for ms


# legend


#ax.legend()
#plt.legend()



# annotation / label
ann = []
N = len(df)
for i in range(N):
    ann.append(ax.annotate(df.label[i], xy = (df.x[i],df.y[i])))
mask = np.zeros(fig.canvas.get_width_height(), bool)

plt.tight_layout()
#overlapping labels removal
fig.canvas.draw()
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
plt.show()
