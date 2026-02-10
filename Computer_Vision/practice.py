import cv2
import numpy as np

cap = cv2.VideoCapture(0)
canvas = None


while True:
    ret, frame = cap.read()
    if not ret: break
    
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(img)
    
    cv2.imshow('Original Frame', frame)
    cv2.waitKey(1)
    cv2.imshow('Gray Frame', img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()