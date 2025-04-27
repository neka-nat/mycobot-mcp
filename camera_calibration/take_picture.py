import os
import cv2
import threading


if_save = False
# Set the camera number (due to different computer models, the number assigned to the USB camera may be different, usually 0 or 1)
cap_num = int(input("Input the camare number:"))
# Set the name of the stored picture to 1, which means that it is accumulated and stored from 1. For example: 1.jpg, 2.jpg, 3.jpg.....
name = int(input("Input start name, use number:"))

cap = cv2.VideoCapture(cap_num)
dir_path = os.path.dirname(__file__)


def save():
    global if_save
    while True:
        input("Input any to save a image:")
        if_save = True

# Start the thread for camera shooting
t = threading.Thread(target=save)
# Set to run asynchronously
t.setDaemon(True)
t.start()

while cv2.waitKey(1) != ord("q"):
    _, frame = cap.read()
    if if_save:
        # Set the name to the current path, otherwise the storage location will change due to the running environment
        img_name = os.path.join(dir_path,str(name)+".jpg") 
        # Store pictures
        cv2.imwrite(img_name, frame)
        print("Save {} successful.".format(img_name))
        name += 1
        if_save = False
    cv2.imshow("", frame)
