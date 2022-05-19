import cv2
import imutils
import numpy as np
import pytesseract
from imutils.video import VideoStream
from PIL import Image
from picamera.array import PiRGBArray
import os
import time
from firebase import Firebase
import RPi.GPIO as GPIO
from datetime import datetime

servoPIN = 17
ledR = 27
ledY = 22
ledG = 23
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(servoPIN, GPIO.OUT)
GPIO.setup(ledR,GPIO.OUT)
GPIO.setup(ledY,GPIO.OUT)
GPIO.setup(ledG,GPIO.OUT)
p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz
p.start(7.5) 
time.sleep(2)
p.ChangeDutyCycle(0)
config = {
  "apiKey": "",
  "authDomain": "",
  "databaseURL": "
  "storageBucket": ""
}

firebase = Firebase(config)
db = firebase.database()

camera = cv2.VideoCapture(0)
camera.set(3, 640)  
camera.set(4, 480)  

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')  
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX


id = 1  

names = ['', 'Tom']  
plateNum = ''

cam = cv2.VideoCapture(2)
cam.set(3, 640)  
cam.set(4, 480)  

# recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
con = 0
startt = 0
unknown = False
finishface = 0
count = 0

def plateee():
        global con
        global startt
        global plateNum
        global name
        
        finishh = startt + 10
        ret1, image = camera.read()
        if ret1:
                cv2.imshow("Frame", image)
                key = cv2.waitKey(1) & 0xFF
                
                if time.time () > finishh :                  
                     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
                     gray = cv2.bilateralFilter(gray, 11, 17, 17) 
                     edged = cv2.Canny(gray, 30, 200) 
                     cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                     cnts = imutils.grab_contours(cnts)
                     cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
                     screenCnt = None
                     for c in cnts:
                        peri = cv2.arcLength(c, True)
                        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                        if len(approx) == 4:
                          screenCnt = approx
                         
                     if screenCnt is None:
                       detected = 0
                       print ("No contour detected")
                       con = 2
                     else:
                       detected = 1
                     if detected == 1:
                       cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
                       mask = np.zeros(gray.shape,np.uint8)
                       new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
                       new_image = cv2.bitwise_and(image,image,mask=mask)
                       (x, y) = np.where(mask == 255)
                       (topx, topy) = (np.min(x), np.min(y))
                       (bottomx, bottomy) = (np.max(x), np.max(y))
                       Cropped = gray[topx:bottomx+1, topy:bottomy+1]
                       text = pytesseract.image_to_string(Cropped, config='--psm 11')
                       text =''.join(filter(str.isalnum, text)) 
                       print("Detected Number is:",text)
                       if text == plateNum:
                               GPIO.output(ledR,GPIO.LOW)
                               GPIO.output(ledY,GPIO.LOW)
                               GPIO.output(ledG,GPIO.HIGH)
                               print("pass")
                               p.ChangeDutyCycle(12.5)
                               time.sleep(1)
                               p.ChangeDutyCycle(0)
                               now = datetime.now()
                               dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                               data = name + " " + plateNum + " at " + dt_string
                               db.child("history").push(data)
                               
                       else:
                                GPIO.output(ledR,GPIO.HIGH)
                                GPIO.output(ledY,GPIO.LOW)
                                GPIO.output(ledG,GPIO.LOW)
                                print("not pass")
                                
        ##             cv2.imshow("Frame", image)
        ##             cv2.imshow('Cropped',Cropped)
        ##             cv2.waitKey(0)

                       con = 2

def faceee():
    global con
    global startt
    global plateNum
    global name
    global unknown
    global finishface
    global count
    
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

       
        if (confidence < 70):
            unknown = False
            GPIO.output(ledR,GPIO.LOW)
            GPIO.output(ledY,GPIO.HIGH)
            GPIO.output(ledG,GPIO.LOW)
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            startt =  time.time ()
            name = str(id)
            plateNum = db.child(str(id) + "/plate").get()
            plateNum = plateNum.val()
            print(plateNum)
            con = 1
        else:
#               if unknown == False: 
#                       startface = time.time ()
#                       finishface = startface + 10
#                       unknown = True
#                       print("facestart")
#               else:                                               
#                       if time.time () > finishface:
#                               count += 1
#                               print("faceend")
#                               unknown = False
#                               now = datetime.now()
#                               timee = now.strftime("%d/%m/%Y %H:%M:%S")
#                               cv2.imwrite("unknown/" +str(count)+ ".jpg", img)
                                     
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

def camer():
        ret, img = cam.read()
        cv2.imshow('camera', img)
        

while True:
   
    if con == 0:
        GPIO.output(ledR,GPIO.HIGH)
        GPIO.output(ledY,GPIO.LOW)
        GPIO.output(ledG,GPIO.LOW)
        faceee()
    elif con == 1:
        plateee()
    else:
        
        start = time.time ()
        finish = start + 5
        while time.time () < finish : 
               camer()
        p.ChangeDutyCycle(7.5)
        time.sleep(1)
        p.ChangeDutyCycle(0)
        con = 0;
        
             
    k = cv2.waitKey(10) & 0xff  
    if k == 27:
        break

camera.release()
cam.release()
cv2.destroyAllWindows()
                 