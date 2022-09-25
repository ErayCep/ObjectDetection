import cv2
import numpy as np
board = cv2.imread('images/flat_chessboard.png')
gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

dst = cv2.dilate(dst, None)

board[dst> 0.01*dst.max()] = [0,0,255]
cv2.namedWindow('harris')

while True:
    cv2.imshow('harris', board)

    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
