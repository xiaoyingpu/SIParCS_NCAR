import cv2
import numpy as np
import sys

if len(sys.argv) != 2:
    print("frameworkpython <fname> <img name>")
    exit()

img = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #why
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)

img=cv2.drawKeypoints(gray,kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite(str(sys.argv[1]) + "sift_kp.jpg", img)
