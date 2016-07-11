from netCDF4 import Dataset
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

N = len(aice[:,0,0])        # 1700+ timestamps

# need timestamps instead of indices


for i in range(N):






fh.close()
