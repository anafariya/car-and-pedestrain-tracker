import cv2

#our image
img_file= 'car.jpg'

#our pre trained car classifier
classifier_file='car_detection.xml'

#create opencv image
img=cv2.imread(img_file)

#convert to bnw for haarcascade
bnw=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#our pre trained car classifier
car_tracker=cv2.CascadeClassifier(classifier_file)

#detect cars
carss=car_tracker.detectMultiScale(bnw)

#draw rectangle
for(x,y,w,h) in carss:

   cv2.rectangle(img, (x,y),(x+w, y+h), (0,0,255),2)

#show the image with car spotted
cv2.imshow('ana car',img)

cv2.waitKey()



