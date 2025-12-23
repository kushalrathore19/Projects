import cv2
import numpy as np

# Initialise CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
# Start video capture
cap = cv2.VideoCapture(0)

#view histogram function
def get_histogram(image):
    #calculate histogram
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    #a black image to draw the histogram (using numpy)
    hist_img = np.zeros((256, 256, 3), dtype=np.uint8)
    
    #normalize the histogram to fit in the hist_img height
    cv2.normalize(histogram, histogram, alpha=0, beta=hist_img.shape[0], norm_type=cv2.NORM_MINMAX)
    
    #draw the histogram
    for i in range(1, 256):
        cv2.line(hist_img, (i-1, 256 - int(histogram[i-1])), (i, 256 - int(histogram[i])), (255, 255, 255), 1)
    return histogram
#function ends here




#starting loop
while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break
    
    #histogram view
    #convert to gray scale for histogram
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #get histogram view
    hist_view = get_histogram(gray)
    #resize histogram view
    hist_view = cv2.resize(hist_view, (320, 240))
    
    
    
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
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Night Vision Filter', enhanced_frame)
    
    
    
    
    # to end the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# end 
cap.release()
cv2.destroyAllWindows()