import cv2
import imutils
import numpy as np
import pytesseract
from imutils.video import VideoStream
from PIL import Image
from picamera.array import PiRGBArray

camera = cv2.VideoCapture(0)
##camera.set(3, 640)  # set video widht
##camera.set(4, 480)  # set video height
#camera.resolution = (640, 480)
#camera.framerate = 30
#rawCapture = PiRGBArray(camera, size=(640, 480))
while True:
        ret,image = camera.read()
#image = imutils.resize(image, width=500)
        cv2.imshow("Frame", image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("s"):
                
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
                text = ''.join(filter(str.isalnum, text)) 
                print("Detected Number is:",text)
                cv2.imshow("Frame", image)
                cv2.imshow('Cropped',Cropped)
                cv2.waitKey(0)

camera.release()
cv2.destroyAllWindows()
