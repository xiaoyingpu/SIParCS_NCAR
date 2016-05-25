import math, operator
from PIL import Image        # import Image not supported

import sys


def rmsd(img_n1, img_n2):
    """
    img_n1: file name string
    img_n2
    """
    h1 = Image.open(img_n1).histogram()
    h2 = Image.open(img_n2).histogram()
    print("size of historgram 1: {} & {}".format(len(h1), len(h2)))
    rms = math.sqrt(reduce(operator.add, map(lambda a,b: (a-b) ** 2, h1, h2))/len(h1))

    return rms

def main():
    if len(sys.argv) != 3:
        print("Usage: frameworkpython rmsd_isomap.py img1 img2")
        exit()

    print(rmsd(sys.argv[1], sys.argv[2]))

main()
