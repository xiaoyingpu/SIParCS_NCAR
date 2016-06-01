import os, sys
from subprocess import call

f_list = []
if not len(sys.argv) == 4:
    print("Usage: python gdal_translate.py <nc_dir> <tif_dir> <variable name>")
    exit()


for f in os.listdir(sys.argv[1]):
    if f.endswith(".nc"):
        f_list.append(f)


os.chdir(sys.argv[1])
for i in range(len(f_list)):

    c = ["gdal_translate"]
    c.append("NETCDF:\"{}\":{}".format(f_list[i], sys.argv[3]))     #variable name
    # use absolute path for the output tiff
    tif_path = os.path.abspath(sys.argv[2])
    if not os.path.exists(tif_path):
        os.makedirs(tif_path)
    if not tif_path.endswith("/"):
        tif_path += "/"

    tif_path += "{}-{}.tif".format(f_list[i], sys.argv[3])  # add file name
    c = c + ["-scale", "-ot", "byte", "-of", "gtiff", tif_path]
    print(c)
    call(c)




