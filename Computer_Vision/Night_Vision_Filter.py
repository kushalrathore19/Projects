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
    return hist_img
#function ends here

# Defining font and style
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
color = (255, 255, 255) # White font color
thickness = 2

# Create a named window for fullscreen display
cv2.namedWindow('Night Vision Filter with Histogram', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Night Vision Filter with Histogram', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

#starting loop
while True:
    # Read frame from camera
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    half_h = h // 2
    side_w = w // 3
    
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
    #displaying the resulting frame (1/2 of the frame)
    #resize to half the frame
    enhanced_frame = cv2.resize(enhanced_frame, (w - side_w, h))

    
    
        
    #histogram view (1/4 of the frame)
    #convert to gray scale for histogram
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #get histogram view
    hist_view = get_histogram(gray)
    #resize histogram view
    hist_view = cv2.resize(hist_view, (side_w, half_h))
    
    
    
    #normal camera view (1/4 of the frame)
    normal = cv2.resize(frame, (side_w, half_h))
    
    #add texts to each view
    cv2.putText(enhanced_frame, 'Filtered View', (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(normal, 'Normal View', (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
    cv2.putText(hist_view, 'Histogram View', (10, 30), font, font_scale, color, thickness, cv2.LINE_AA)
    
    
    

    # STACKING WITH NUMPY 
    # Stack the two small ones vertically (Normal on top, Histogram on bottom)
    right_column = np.vstack((normal, hist_view))
    
    # Combining the large Filtered view with the Right Column horizontally
    dashboard = np.hstack((enhanced_frame, right_column))
    
    #showing
    cv2.imshow('Night Vision Filter with Histogram', dashboard)
    # to end the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# end 
cap.release()
cv2.destroyAllWindows()