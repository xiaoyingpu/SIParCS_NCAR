import datetime
from netCDF4 import Dataset, num2date
from skimage.measure import compare_ssim as ssim
import os


def get_datetime(d):
    """
    [YYYYMM]
    """
    d = str(int(d))
    yyyy = int(d[:4])
    mm = int(d[4:])
    return datetime.date(year= yyyy, month = mm, day = 1)




var = "sic"
path = "/Users/puxiaoadmin/cmip5/timeseries"
Walsh = "walsh_chapman.NH.seaice.187001-201112.nc"
NASA = "seaice_conc_monthly_nh_NASA_Bootstrap_v2.nsidc.v02r00.197811-201412.nc"


# change working dir
os.chdir(path)

fh = Dataset(Walsh, mode="r")

aice = fh.variables[var][:] # now numpy array
aice = aice.filled(fill_value = 0)


# ----------get date time-------------
nctime = fh.variables["time"][:]
#try:
#    t_unit = fh.variavles["time"].units
#    calendar = fh.variables["time"].calendar
#except AttributeError:
#    calendar = u"gregorian"
#    t_unit = "YYYYMM"
#date_var = [].append((num2date(nctime, units = t_unit, calendar = calendar)))




N = len(aice[:,0,0])        # 1700+ timestamps

# the movie analogy

for i in range(1, N):
    date = get_datetime( nctime[i])
    cur_frame = aice[i,:,:]
    dist = ssim(cur_frame, prev_frame)
    prev_frame = cur_frame


fh.close()
