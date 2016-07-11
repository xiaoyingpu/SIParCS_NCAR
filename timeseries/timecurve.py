import datetime
import numpy as np
from netCDF4 import Dataset, num2date
from skimage.measure import compare_ssim as ssim
import os, math, itertools, json


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

old_path = os.path.abspath(".")

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
N = 12                      # start small

# create time labels
time_label = []
for i in range(1, N):       # change this
    date = get_datetime(nctime[i])
    date_str = date.isoformat() + " 00:00:00.0" # I don't care
    time_label.append(date_str)
    print date_str


# create distance matrix
dm = np.ones((N, N))
for tup in itertools.combinations(range(N), 2):
    i, j = tup
    s = ssim(aice[i,:,:], aice[j,:,:])
    dm[i][j] = s
    dm[j][i] = s



json_dic = {}
print type(dm)
json_dic["distancematrix"] = dm.tolist()
json_dic["data"] = {}
json_dic['data']['name'] = "1870"
json_dic['data']['timelabels'] = time_label

os.chdir(old_path)

with open("out.json", "w") as f:
    json.dump(json_dic, f, indent = 4, sort_keys = True)


fh.close()
