import cv2 as cv
from matplotlib import pyplot as plt

#gambar asli
img = plt.imread("foto.jpg") #mengambil gambar dalam mode matplotlib

#gambar asli di detek wajahnya
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml") #module deteksi wajah
hasildetekawal = plt.imread("foto.jpg") 
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4) #untuk menemukan wajah atau mata

for(x,y,w,h) in faces:
    cv.rectangle(img, (x,y), (x+w, y+h), (255, 0,0), 2) #proses deteksi wajah

#pengaturan brightness & kontras
hasilimageenhanc = cv.convertScaleAbs(img, alpha=1.8, beta=30)

#deteksi tepi
gray = cv.cvtColor(hasilimageenhanc, cv.COLOR_BGR2GRAY)
garistepi = cv.Canny(img,40,55,apertureSize =3) #image asli
garistepifinal = cv.Canny(hasilimageenhanc,40,55,apertureSize =3) #image setelah birghtness & kontras

#gambar akhir di detek wajahnya
gambarfinal =  cv.convertScaleAbs(img, alpha=1.8, beta=30) #mengambil gambar imageenhanc
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_alt.xml") #module deteksi wajah
gray = cv.cvtColor(gambarfinal, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for(x,y,w,h) in faces:
    cv.rectangle(gambarfinal, (x,y), (x+w, y+h), (255, 0,0), 2) #proses deteksi wajah


titles = ['Asli','deteksi tepi','deteksi wajah','Brightness&Contrast','deteksi tepi','deteksi wajah']
images = [img, garistepi, hasildetekawal,  hasilimageenhanc, garistepifinal, gambarfinal]

for i in range(6): #mengambil 6 gambar
    plt.subplot(2,3,i+1),plt.imshow(images[i], cmap="gray") #menempatkan dan memunculkan gambar di matplotlib
    plt.title(titles[i]) #memberi judul pad gambar
    plt.xticks([]),plt.yticks([]) #menghasil koordinat sumbu x dan y


plt.show()

cv.waitKey(0)
