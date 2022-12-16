import cv2
image = cv2.imread("foto.jpg")
hasil = cv2.convertScaleAbs(image, alpha=1.8, beta=30)
gray = cv2.cvtColor(hasil, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,40,55,apertureSize =3)
cv2.imshow("edges in", edges)
cv2.waitKey(0)