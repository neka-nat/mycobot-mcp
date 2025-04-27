import os
import numpy as np
import cv2 as cv


if __name__ == "__main__":
    # Load calibration parameters
    load = np.load(os.path.join(os.path.dirname(__file__), "mtx_dist.npz"))
    mtx = load["mtx"]
    dist = load["dist"]

    cap_num = int(input("Input camera number:"))
    cap = cv.VideoCapture(cap_num)
    while cv.waitKey(1) != ord("q"):
        _, img = cap.read()
        h, w = img.shape[:2]
        # Camera calibration
        dst = cv.undistort(img, mtx, dist, None)
        merge = np.hstack((img, dst))
        cv.imshow("", merge)
