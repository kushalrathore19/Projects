import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

# Load image
img = cv2.imread("badQuality.jpg", cv2.IMREAD_GRAYSCALE)

# ---------- NIGHT VISION (CLAHE) ----------
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
filtered = clahe.apply(img)

# ---------- HISTOGRAM + CDF ----------
fig = plt.Figure(figsize=(4, 4))
canvas = FigureCanvasAgg(fig)
ax = fig.add_subplot(111)

hist = cv2.calcHist([img], [0], None, [256], [0, 256])
cdf = hist.cumsum()
cdf_norm = cdf * hist.max() / cdf.max()

ax.plot(hist, color='black')
ax.plot(cdf_norm, color='blue')
ax.set_title("Histogram + CDF")
ax.set_xlabel("Pixel Intensity")
ax.set_ylabel("Count")

canvas.draw()

# Convert plot to image
graph = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
graph = graph.reshape(canvas.get_width_height()[::-1] + (3,))
graph = cv2.cvtColor(graph, cv2.COLOR_RGB2BGR)

# ---------- RESIZING ----------
H, W = img.shape

filtered = cv2.resize(filtered, (W, H))
normal = cv2.resize(img, (W//2, H//2))
graph = cv2.resize(graph, (W//2, H//2))

# Convert grayscale to BGR for stacking
filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
normal = cv2.cvtColor(normal, cv2.COLOR_GRAY2BGR)

# ---------- COMBINE LAYOUT ----------
top = filtered
bottom = np.hstack((normal, graph))
final = np.vstack((top, bottom))

# ---------- DISPLAY ----------
cv2.imshow("Night Vision Lab Output", final)
cv2.waitKey(0)
cv2.destroyAllWindows()
