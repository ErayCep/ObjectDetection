import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('images/sammy_face.jpg')
blur = cv2.blur(img, ksize=(5,5))

med_value = np.median(img)

upper = int(min(255, 1.3*med_value))
lower = int(max(0, 0.7*med_value))

edges = cv2.Canny(image=blur, threshold1 = lower, threshold2=upper+50)
plt.imshow(edges)
plt.show()
