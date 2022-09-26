import cv2
import numpy as np

real_chess = cv2.imread('images/real_chessboard.jpg')
gray = cv2.cvtColor(real_chess, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

real_chess[dst>0.01*dst.max()] = [0,0,255]
cv2.namedWindow('chess')

while True:
    cv2.imshow('chess', real_chess)

    if cv2.waitKey(20) & 0xFF == 27:
        break

cv2.destroyAllWindows()
