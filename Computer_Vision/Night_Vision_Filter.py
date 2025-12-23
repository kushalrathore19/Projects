import cv2
import numpy as np

# Initialise CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# Start video capture
cap = cv2.VideoCapture(0)

#starting loop
while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break
    #conver to LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    #split into LAB = (l, a, b) channels
    l, a, b = cv2.split(lab)
    
    #apply CLAHE to l-channel only
    cl = clahe.apply(l)
    
    #merge enhanced l-channel with a and b channels
    enhanced_lab = cv2.merge((cl, a, b))
    
    #convert back to BGR color space
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    #display the resulting frame
    cv2.imshow('Night Vision Filter', enhanced_frame)
    
    # to end the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# end 
cap.release()
cv2.destroyAllWindows()