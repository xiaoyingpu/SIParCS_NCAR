# input netCDF files
# output x, y coordinates -> csv for visualization

from netCDF4 import Dataset
import numpy as np


wd = "~/cmip5/sea_ice/"

os.chdir(wd)

fname = "NASA_Bootstrap_v2_NH.cvdp_data.1979-2005.nc"

fh = Dataset(fname, mode="r")

aice = fh.variables["aice_nh_spatialmean_ann"]


fh.close()

print aice
