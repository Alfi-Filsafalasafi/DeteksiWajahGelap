import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
imgc = cv2.imread("D.png")
img = cv2.cvtColor(imgc, cv2.COLOR_BGR2GRAY)
 
w_k = np.array([[0, -1, 0],
[-1,5, -1],
[0, -1, 0],],
dtype='float')
 
w_k = np.rot90(w_k, 2)
 
print (img.shape, w_k.shape)
f = signal.convolve2d(img, w_k, 'valid')
print(np.min(f))
 
plt.subplot(121),plt.imshow(imgc),plt.title('original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(f,'gray'),plt.title('gaussian Blur')
plt.xticks([]), plt.yticks([])
plt.show()