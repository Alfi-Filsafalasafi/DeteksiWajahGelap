import cv2  as cv

test1 = cv.imread("cobanih.png")
gray_img = cv.cvtColor(test1, cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray_img)

cv.waitKey(0)
