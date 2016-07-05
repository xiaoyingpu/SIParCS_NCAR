# input netCDF files
# output x, y coordinates -> csv for visualization

from netCDF4 import Dataset
import numpy as np
import os

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def get_data(wd, fname, var):

    os.chdir(wd)

    fh = Dataset(fname, mode="r")
    aice_nh_spatialmean_ann = fh.variables[var]
    aice = aice_nh_spatialmean_ann[:]
    lons = fh.variables["lon"][:]
    lats = fh.variables["lat"][:]
    fh.close()
    return aice

var = "aice_nh_spatialmean_ann"
fname = "NASA_Bootstrap_v2_NH.cvdp_data.1979-2005.nc"
wd = "/Users/puxiaoadmin/cmip5/sea_ice"

l = get_data(wd, fname, var)
