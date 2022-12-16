import cv2 as cv
import numpy as np
from scipy import signal
img = cv.imread("cobanih.png")
hasil = cv.convertScaleAbs(img, alpha=1.8, beta=30)
cv.imshow("hasil", hasil)
cv.imshow("cobanih",img)

cv.waitKey(0)
