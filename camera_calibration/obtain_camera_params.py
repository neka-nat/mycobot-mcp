import os
import glob
import numpy as np
import cv2 as cv
from pprint import pprint


def calibration_camera(row, col, path=None, cap_num=None, saving=False):
    """Calibrate camera

    Parameter Description:
        row (int): the number of rows in the grid.
        col (int): the number of columns in the grid.
        path (string): the location where the calibration picture is stored.
        cap_num (int): indicates the number of the camera, usually 0 or 1
        saving (bool): whether to store the camera matrix and distortion coefficient (. npz)
    """

    # Termination criteria / failure criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # Prepare object points, such as (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    obj_p = np.zeros((row * col, 3), np.float32)
    obj_p[:, :2] = np.mgrid[0:row, 0:col].T.reshape(-1, 2)
    # Groups are used to store object points and image points from all images.
    obj_points = []  # The position of 3D points in the real world.
    img_points = []  # Position of 2D point in the picture.

    gray = None

    def _find_grid(img):
        # Use parameters outside the function
        nonlocal gray, obj_points, img_points
        # Convert picture to gray picture
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Look for the corner of the chessboard
        ret, corners = cv.findChessboardCorners(gray, (row, col), None)
        # If found, the processed 2D points and 3D points are added
        if ret == True:
            obj_points.append(obj_p)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_points.append(corners)
            # Draw and show the corner found in the picture
            cv.drawChessboardCorners(img, (row, col), corners2, ret)

    # It is required that you must select one of image calibration or camera real-time capture calibration
    if path and cap_num:
        raise Exception("The parameter `path` and `cap_num` only need one.")
    # Picture calibration
    if path:
        # Get all pictures in the current path
        images = glob.glob(os.path.join(path, "*.jpg"))
        pprint(images)
        # Process each acquired picture
        for f_name in images:
            # Read picture
            img = cv.imread(f_name)
            _find_grid(img)
            # Show pictures
            cv.imshow("img", img)
            # Picture display wait for 0.5s
            cv.waitKey(500)
    # Camera real-time capture calibration
    if cap_num:
        # Turn on the camera
        cap = cv.VideoCapture(cap_num)
        while True:
            # Read every picture after the camera is turned on
            _, img = cap.read()
            _find_grid(img)
            cv.imshow("img", img)
            cv.waitKey(500)
            print(len(obj_points))
            if len(obj_points) > 14:
                break
    # Destroy display window
    cv.destroyAllWindows()
    # The camera matrix and distortion coefficient are obtained by calculating the obtained 3D points and 2D points
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        obj_points, img_points, gray.shape[::-1], None, None
    )
    print("ret: {}".format(ret))
    print("matrix:")
    pprint(mtx)
    print("distortion: {}".format(dist))
    # Decide whether to store the calculated parameters
    if saving:
        np.savez(os.path.join(os.path.dirname(__file__), "mtx_dist.npz"), mtx=mtx, dist=dist)

    mean_error = 0
    for i in range(len(obj_points)):
        img_points_2, _ = cv.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(img_points[i], img_points_2, cv.NORM_L2) / len(img_points_2)
        mean_error += error
    print("total error: {}".format(mean_error / len(obj_points)))

    return mtx, dist


if __name__ == "__main__":
    path = os.path.dirname(__file__)
    mtx, dist = calibration_camera(8, 6, path, saving=True)
    # Set whether the calculated parameters need to be tested
    if_test = input("If testing the result (default: no), [yes/no]:")
    if if_test not in ["y", "Y", "yes", "Yes"]:
        exit(0)

    cap_num = int(input("Input camera number:"))
    cap = cv.VideoCapture(cap_num)
    while cv.waitKey(1) != ord("q"):
        _, img = cap.read()
        h, w = img.shape[:2]
        # Camera calibration
        dst = cv.undistort(img, mtx, dist)
        cv.imshow("", dst)
