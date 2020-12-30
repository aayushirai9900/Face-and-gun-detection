import numpy as np
import cv2
import imutils

#Function for detecting gun

def gun(frame,gun_cascade,firstFrame,gun_exist):
    if ret:
         
        frame = imutils.resize(frame, width = 500) 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        gun = gun_cascade.detectMultiScale(gray,5, 5, minSize = (100, 100))
    
       
        if len(gun) > 0: 
            gun_exist = True
            
        for (x, y, w, h) in gun: 
            
            frame = cv2.rectangle(frame, 
                                (x, y), 
                                (x + w, y + h), 
                                (255, 0, 0), 2) 
            roi_gray = gray[y:y + h, x:x + w] 
            roi_color = frame[y:y + h, x:x + w]     
            
        if firstFrame is None: 
            firstFrame = gray 
            #continue
            
    
        # print(datetime.date(2019)) 
        # draw the text and timestamp on the frame 
        cv2.putText(frame, str("gun"), 
                    (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.35, (0, 0, 255), 1)
        cv2.imshow("Weapon Detector",frame) 
#Function for detecting face

def face(detector,frame):
    if ret:
        faces = detector.detectMultiScale(frame,1.1,4)

        for face in faces:
            x, y, w, h = face
            
            cut = frame[y:y+h, x:x+w]

            fix = cv2.resize(cut, (100, 100))
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)
            out = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.imshow("My Face", frame)
            
#Driver code
gun_cascade = cv2.CascadeClassifier('./cascade.xml') 
detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0) 
   
firstFrame = None
gun_exist = False
while True :
    ret, frame = cap.read()

    gun(frame,gun_cascade,firstFrame,gun_exist)
    face(detector,frame)

    # cv2.imshow("xyz", frame) 
    key = cv2.waitKey(1) 

    if key == ord("q"):
        break 

cap.release()
cv2.destroyAllWindows()
