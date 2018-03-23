import numpy as np 
import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load our image and then convert it to grayscale
image = cv2.imread('sixfaces410.jpg',0)

faces = face_classifier.detectMultiScale(image,1.3,5)

if faces is ():
	print("No faces found")

for (x,y,w,h) in faces:
	cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
	cv2.imshow('face',image)
	cv2.waitKey(0)

	eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')
	face_image = image[y:y+h,x:x+w]
	cv2.imshow('cropped',face_image)
	cv2.waitKey(0)

	eyes = eye_classifier.detectMultiScale(face_image)

	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(face_image,(ex,ey),(ex+ew,ey+eh),(0,127,255),1)
		cv2.imshow('eyes',face_image)
		cv2.waitKey(0)

cv2.destroyAllWindows()