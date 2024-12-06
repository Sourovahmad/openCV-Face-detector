import cv2 as cv


img = cv.imread("./image1.jpg") #read the image.


# convert to Graysycl
gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


face_classifier = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
)

for (x, y, w, h) in face:
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

cv.imwrite('gray_image.jpg', img_rgb)

