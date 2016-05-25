import numpy as np
import cv2
import math, operator
from PIL import Image        # import Image not supported
import sys



def rmsd_histogram(img_n1, img_n2):
    """
    img_n1: file name string
    img_n2
    """
    h1 = Image.open(img_n1).histogram()
    h2 = Image.open(img_n2).histogram()
    print("size of historgram 1: {} & {}".format(len(h1), len(h2)))
    rms = math.sqrt(reduce(operator.add, map(lambda a,b: (a-b) ** 2, h1, h2))/len(h1))

    return rms


def mse(filename1, filename2):
    """
    img1 and img2 have the same dimension
    """
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img2.shape[1])
    return err



def main():
    if len(sys.argv) != 3:
        print("Usage: frameworkpython rmsd_isomap.py img1 img2")
        exit()

    print(mse(sys.argv[1], sys.argv[2]))

main()
