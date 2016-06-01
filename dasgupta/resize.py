# resize all images in dir
# with specified (width, height)

import sys, os
from PIL import Image

if len(sys.argv) != 4:
    print("Usage: frameworkpython resize.py dir/ width hight")
    exit()


w = int(sys.argv[2])
h = int(sys.argv[3])
print(w,h)
# change working directory
os.chdir(sys.argv[1])

f_list = []
for f in os.listdir("."):
    if not f.endswith("tif"):
        continue
    f_list.append(f)

    im = Image.open(f)
    im = im.resize([w,h],Image.ANTIALIAS)
    im.save("r-{}".format(f))
