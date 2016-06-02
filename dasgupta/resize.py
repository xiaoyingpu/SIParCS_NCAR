# resize all images in dir
# with specified (width, height)

import sys, os
from PIL import Image

if len(sys.argv) != 4:
    print("Usage: frameworkpython resize.py dir/ width hight")
    exit()



f_list = []
w = int(sys.argv[2])
h = int(sys.argv[3])

# change working directory
os.chdir(sys.argv[1])

if not os.path.exists("./resize"):
    os.makedirs("./resize")


for f in os.listdir("."):
    if not f.endswith("tif"):
        continue
    f_list.append(f)

    im = Image.open(f)
    im = im.resize([w,h],Image.ANTIALIAS)
    im.save("./resize/r-{}".format(f))
