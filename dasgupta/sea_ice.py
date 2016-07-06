# input netCDF files
# output x, y coordinates -> csv for visualization

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

# replace the kNN, graph approach with a distance matrix,
# SSIM as the distance metric
# make sense????

import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

from netCDF4 import Dataset

var = "aice_nh_spatialmean_ann"  # this thing is global, right?

#fname = "ACCESS1-0.cvdp_data.1900-2005.nc.reggrided.nc"
#wd = "/Users/puxiaoadmin/cmip5/sea_ice/regridded"


def get_data(fname):

    fh = Dataset(fname, mode="r")
    aice_nh_spatialmean_ann = fh.variables[var]
    aice = aice_nh_spatialmean_ann[:]
    aice = aice.filled(fill_value = 0)
    fh.close()
    return aice



def get_model(fname):
    return fname.split("_", 1)[0].split("-",1)[0]


def test():
    d = "/Users/puxiaoadmin/cmip5/sea_ice/regridded"
    os.chdir(d)
    f = "Walsh_and_Chapman.cvdp_data.1979-2005.nc.reggrided.nc"
    l = get_data(f)
    print(type(l))
    print(l[0])

def main():
    """
    Usage: frameworkpyton sea_ice.py <.nc dir>
    """


    # get list of files
    script_path = os.path.abspath(".")

    if len(sys.argv) != 2:
        print("need dir arg")
        exit()
    nc_dir = os.path.abspath(sys.argv[1])
    f_list = []
    for f in os.listdir(nc_dir):
        if f.endswith(".nc"):
            f_list.append(f)

    N = len(f_list)


    # change dir and compute ssim
    os.chdir(nc_dir)
    dm = np.zeros((N,N))
    for tup in itertools.combinations(range(N),2):
        i, j = tup
        i_dat = get_data(f_list[i])
        j_dat = get_data(f_list[j])
        s = 1 - ssim(i_dat, j_dat)
        dm[i][j] = s
        dm[j][i] = s

    # MDS
    Y = manifold.MDS(n_components=2, dissimilarity='precomputed').fit_transform(dm)
    Y = np.array(Y)

    print Y

    # save output
    f_persist = "cmip5_4.0.0_{}.csv".format(var)
    os.chdir(os.path.join(script_path, "csv"))
    with open(f_persist, "w+") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y","label", "fname"])
        for i in range(N):
            row = [Y[:,0][i], Y[:,1][i], get_model(f_list[i]), f_list[i]]
            writer.writerow(row)
#test()
main()

