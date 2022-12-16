import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
imgawal = cv2.imread("foto.png")
hasil = cv2.convertScaleAbs(imgawal, alpha=1.8, beta=30)

gray = cv2.cvtColor(hasil, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for(x,y,w,h) in faces:
    cv2.rectangle(hasil, (x,y), (x+w, y+h), (255, 0,0), 2)

cv2.imshow('img', hasil)
cv2.imshow('img awal', imgawal)
cv2.imshow('img awal di detek wajahnya', imgawal)
cv2.waitKey()