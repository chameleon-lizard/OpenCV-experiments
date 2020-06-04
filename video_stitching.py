import numpy as np
import argparse
import imutils
import cv2

# Getting the args
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--left", type=str, required=True,
                help="path to the left video")
ap.add_argument("-r", "--right", type=str, required=True,
                help="path to the right video")
ap.add_argument("-m", "--method", type=str, required=True,
                help="method of denoising")

args = vars(ap.parse_args())

# Getting the method of denoising from arguments
method = args["method"]

# Creating VideoCapture streams for left and right part of video
left = cv2.VideoCapture(args['left'])
right = cv2.VideoCapture(args['right'])

# Creating the stitcher
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
total = 0

while True:
    # Getting the frames from videos
    ret1, left_frame = left.read()
    ret2, right_frame = right.read()

    # If we reached the end of one of the streams,
    if ret1 == False or ret2 == False:
        break

    # Stitching the frames
    (status, stitched) = stitcher.stitch([left_frame, right_frame])

    # Checking the status
    if status == 1:
        print("[RESULT] image stitching failed (not enough images)")
        continue
    elif status == 2:
        print("[RESULT] image stitching failed (failed to estimate homography matrix)")
        continue
    elif status == 3:
        print("[RESULT] image stitching failed (failed to adjust camera parameters)")
        continue
    elif status == 0:
        if method == "Bilateral":
        	stitched = cv2.bilateralFilter(stitched, 9, 50, 50)
        elif method == "Gaussian":
        	stitched = cv2.GaussianBlur(stitched, (3, 3), cv2.BORDER_DEFAULT)
        elif method == "FastNl":
        	stitched = cv2.fastNlMeansDenoisingColored(stitched, hColor=30)
        elif method == "none":
        	pass

    # Resizing the stitched frame
    stitched = imutils.resize(stitched, 1280)

    total += 1
    cv2.imshow("Result", stitched)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
left.release()
right.release()

cv2.destroyAllWindows()
