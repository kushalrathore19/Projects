import cv2
import numpy as np
webcam = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (255, 292, 203) 
thickness = 2

while True:
    ret, frame = webcam.read()
    hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    screen = np.array_equal(hist, np.zeros_like(hist))
    if screen:
        cv2.putText(frame, 'No Signal', (50, 50), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Webcam Feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()