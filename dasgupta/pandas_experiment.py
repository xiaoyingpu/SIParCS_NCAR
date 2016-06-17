import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./csv/ssim_cmip5_with_label.csv")
grp = df.groupby("label")

sns.set_palette("hls", n_colors=25)
fig, ax = plt.subplots()
#palette = np.array(sns.color_palette("hls", 25))
for name, group in grp:
    plt.plot(group.x, group.y, marker = "o",ms = 4, linestyle = " ", label = name)

ax.legend()
plt.show()
