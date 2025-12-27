import cv2
import numpy as np

# 1. LOAD IMAGE
img = cv2.imread("TestImages/test1.jpg")

if img is None:
    print("❌ Image not found. Check path or filename.")
    exit()

orig = img.copy()
h, w = img.shape[:2]

# 2. PREPROCESS
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 50, 150)

# 3. FIND CONTOURS
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

doc = None

# 4. FIND 4-POINT DOCUMENT
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        doc = approx
        break

if doc is None:
    print("❌ No document contour found.")
    exit()

# 5. SOURCE & DESTINATION POINTS
src_pts = np.float32(doc.reshape(4, 2))

dst_pts = np.float32([
    [0, 0],
    [w, 0],
    [w, h],
    [0, h]
])

# 6. PERSPECTIVE TRANSFORM
M = cv2.getPerspectiveTransform(src_pts, dst_pts)
warped = cv2.warpPerspective(orig, M, (w, h))

# 7. ENHANCE (SCAN EFFECT)
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
final_scan = cv2.adaptiveThreshold(
    warped_gray, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)

# 8. VISUALIZE
cv2.drawContours(img, [doc], -1, (0, 255, 0), 2)

cv2.imshow("Original Image", img)
cv2.imshow("Edges", edges)
cv2.imshow("Warped", warped)
cv2.imshow("Scanned Document", final_scan)

cv2.waitKey(0)
cv2.destroyAllWindows()
