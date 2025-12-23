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
    
    # to end the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# end 
cap.release()
cv2.destroyAllWindows()