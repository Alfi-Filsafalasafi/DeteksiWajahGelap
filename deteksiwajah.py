import cv2
import matplotlib.pyplot as plt
import time

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# test1 = cv2.imread("hiii.jpg")
# gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)

haar_face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_alt.xml")
# faces = haar_face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5 )
# print('Faces found: ', len(faces))
# for (x,y,w,h) in faces:
#     cv2.rectangle(test1, (x,y), (x+w, y+h), (0, 255, 0), 2)


def detect_faces(f_cascade, colored_img, scaleFactor =1.1) :
    img_copy = colored_img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img_copy, (x,y), (x+w, y+h), (0, 255, 0), 2)
        return img_copy
    test2 = cv2.imread("GetFoto.jpg")
    faces_detected_img = detect_faces(haar_face_cascade, test2)
    plt.imshow(convertToRGB(faces_detected_img))
