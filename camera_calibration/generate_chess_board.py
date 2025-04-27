import os
import cv2
import numpy as np

# All black picture of 630*890
frame = np.zeros((630, 890, 3), dtype=np.uint8)
row, col, nc = frame.shape

width_of_roi = 90
# Here is the processing of all black pictures, which are separated in black and white zh
for j in range(row):
    data = frame[j]
    for i in range(col):
        f = int(i / width_of_roi) % 2 ^ int(j / width_of_roi) % 2
        if f:
            frame[j][i][0] = 255
            frame[j][i][1] = 255
            frame[j][i][2] = 255

# cv2.imshow("", frame)
# cv2.waitKey(0) & 0xFF == ord("q")
cv2.imwrite(os.path.join(os.path.dirname(__file__), "1.png"), frame)
