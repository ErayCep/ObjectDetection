import cv2
import numpy as np
import matplotlib.pyplot as plt
flat_chess = cv2.imread('images/flat_chessboard.png')
gray = cv2.cvtColor(flat_chess, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 10, 0.01, 10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(gray, (x,y), 4, 255, -1)

plt.imshow(gray, cmap='gray')
plt.show()
print(i)
