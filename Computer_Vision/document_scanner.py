import cv2
import numpy as np


# Function to order points in clockwise order: top-left, top-right, bottom-right, bottom-left
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # Top-left
    rect[2] = pts[np.argmax(s)]   # Bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right
    rect[3] = pts[np.argmax(diff)]  # Bottom-left

    return rect
#load image
img = cv2.imread("TestImages/test1.png")

if img is None:
    exit()

orig = img.copy()
h, w = img.shape[:2]

# preprocess for edge detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)


#find contours
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

doc = None

# finding the document contour
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        doc = approx
        break

if doc is None:
    exit()

# source and destination points for perspective transform
src_pts = order_points(doc.reshape(4, 2))

dst_pts = np.float32([
    [0, 0],
    [w, 0],
    [w, h],
    [0, h]
])

M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(orig, M, (w, h))


# post-process to get a scanned effect
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
final_scan = cv2.adaptiveThreshold(
    warped_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

#draw the document contour on the original image
cv2.drawContours(img, [doc], -1, (0, 255, 0), 2)

cv2.imshow("Original Image", img)
cv2.imshow("Edges", edges)
cv2.imshow("Warped", warped)
cv2.imshow("Scanned Document", final_scan)

cv2.waitKey(0)
cv2.destroyAllWindows()
