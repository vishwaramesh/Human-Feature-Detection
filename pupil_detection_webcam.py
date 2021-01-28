import numpy as np
import cv2
import math
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)
while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eye_list,confidence = eye_cascade.detectMultiScale2(roi_gray)
        print(confidence)
        eye = [eye_list[x] for x in range(len(eye_list)) if confidence[x]>50]
        for (ex,ey,ew,eh) in eye:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            roi_eye = roi_color[ey:ey+eh, ex:ex+ew]
            eye_gray = cv2.cvtColor(~roi_eye, cv2.COLOR_BGR2GRAY)
            ret, thresh_gray = cv2.threshold(eye_gray, 220, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                rect = cv2.boundingRect(contour)
                eye_x, eye_y, width, height = rect
                radius = 0.25 * (width + height)

                area_condition = (50<=area <= 400)
                symmetry_condition = (abs(1 - float(width)/float(height)) <= 0.5)
                fill_condition = (abs(1 - (area / (math.pi * math.pow(radius, 2.0)))) <= 0.55)
                
                print(area_condition,symmetry_condition,fill_condition,area)
                if area_condition and symmetry_condition and fill_condition:
                    cv2.circle(roi_eye, (int(eye_x + radius), int(eye_y + radius)), int(radius), (0,180,0), -1)
    cv2.imshow('img',img)
    if cv2.waitKey(10) == 27:#Use ESC to close the webcam
        break
cap.release()
cv2.destroyAllWindows()