from netCDF4 import Dataset, netcdftime
from skimage.measure import compare_ssim as ssim
import os

var = "sic"
path = "/Users/puxiaoadmin/cmip5/timeseries"
Walsh = "walsh_chapman.NH.seaice.187001-201112.nc"
NASA = "seaice_conc_monthly_nh_NASA_Bootstrap_v2.nsidc.v02r00.197811-201412.nc"


# change working dir
os.chdir(path)

fh = Dataset(Walsh, mode="r")

aice = fh.variables[var][:] # now numpy array
aice = aice.filled(fill_value = 0)


# get dat time
nctime = fh.variables["time"][:]
t_unit = fh.variavles["time"].units
try:
    calendar = fh.variables["time"].calendar
except AttributeError:
    calendar = u"gregorian"

date_var = []
date_var.append(netcdftime.num2date(nctime, units = t_unit, calendar = calendar))


N = len(aice[:,0,0])        # 1700+ timestamps

# need timestamps instead of indices
# the movie analogy
prev_frame = aice[0,:,:]

for i in range(1, N):
    cur_frame = aice[i,:,:]
    dist = ssim(cur_frame, prev_frame)
    prev_frame = cur_frame





fh.close()
