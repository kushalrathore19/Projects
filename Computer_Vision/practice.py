import cv2
import numpy as np
webcam = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (255, 292, 203) 
thickness = 2

while True:
    ret, frame = webcam.read()
    h, w = frame.shape[:2]

    src = np.float32([
        [100, 100],
        [500, 100],
        [550, 400],
        [80, 400]
    ])

    dst = np.float32([
        [0, 0],
        [300, 0],
        [300, 300],
        [0, 300]
    ])
# Perspective matrix
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(frame, M, (300, 300))

    cv2.imshow("Original", frame)
    cv2.imshow("Bird View", warped)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
webcam.release()
cv2.destroyAllWindows()