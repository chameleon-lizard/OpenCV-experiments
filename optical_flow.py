# Importing libraries
import cv2
import numpy as np
import imutils
import argparse
import sys

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

# Creating stitcher object
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()

# Capturing the video file 0 for videocam else you can provide the url
capture_left = cv2.VideoCapture(args['left'])
capture_right = cv2.VideoCapture(args['right'])

# Reading the first frame
status_left, frame1_left = capture_left.read()
status_right, frame1_right = capture_right.read()

# If we reached the end of one of the streams,
if status_left == False or status_right == False:
    sys.exit()

# Denoising if needed
if method == "Bilateral":
    frame1_left = cv2.bilateralFilter(frame1_left, 9, 100, 100)
    frame1_right = cv2.bilateralFilter(frame1_right, 9, 100, 100)
elif method == "Gaussian":
    frame1_left = cv2.GaussianBlur(frame1_left, (3, 3), cv2.BORDER_DEFAULT)
    frame1_right = cv2.GaussianBlur(frame1_right, (3, 3), cv2.BORDER_DEFAULT)
elif method == "FastNl":
    frame1_left = cv2.fastNlMeansDenoisingColored(frame1_left, hColor=30)
    frame1_right = cv2.fastNlMeansDenoisingColored(frame1_right, hColor=30)
elif method == "none":
    pass

# Stitching the frames
(status, prvs) = stitcher.stitch([frame1_left, frame1_right])

# Checking the status
if status == 1:
    print("[RESULT] image stitching failed (not enough images)")
elif status == 2:
    print("[RESULT] image stitching failed (failed to estimate homography matrix)")
elif status == 3:
    print("[RESULT] image stitching failed (failed to adjust camera parameters)")
elif status == 0:
    pass

# Convert to gray scale
prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)

# Till you scan the video
while(1):
    # Capture another frame and convert to gray scale
    status_left, frame2_left = capture_left.read()
    status_right, frame2_right = capture_right.read()

    # If we reached the end of one of the streams,
    if status_left == False or status_right == False:
        break

    # Denoising if needed
    if method == "Bilateral":
        frame2_left = cv2.bilateralFilter(frame2_left, 9, 50, 50)
        frame2_right = cv2.bilateralFilter(frame2_right, 9, 50, 50)
    elif method == "Gaussian":
        frame2_left = cv2.GaussianBlur(frame2_left, (3, 3), cv2.BORDER_DEFAULT)
        frame2_right = cv2.GaussianBlur(frame2_right, (3, 3), cv2.BORDER_DEFAULT)
    elif method == "FastNl":
        frame2_left = cv2.fastNlMeansDenoisingColored(frame2_left, hColor=30)
        frame2_right = cv2.fastNlMeansDenoisingColored(frame2_right, hColor=30)
    elif method == "none":
        pass

    # Stitching the frames
    (status, following) = stitcher.stitch([frame2_left, frame2_right])

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
        pass

    # Converting to greyscale
    following = cv2.cvtColor(following, cv2.COLOR_BGR2GRAY)

    # Resizing the picture to ensure that previous and next pictures are the same size
    following = cv2.resize(following, (len(prvs[0]), len(prvs)), interpolation = cv2.INTER_AREA)

    # Optical flow is now calculated
    flow = cv2.calcOpticalFlowFarneback(prvs, following, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Compute magnitude and angle of 2D vector
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    # Stitcher seems to create different dimension narray, so creating mask from unstitched
    # frame, but resized to size of our stitched picture
    frame2_left = cv2.resize(frame2_left, (len(prvs[0]), len(prvs)), interpolation = cv2.INTER_AREA)
    
    # Create mask
    hsv_mask = np.zeros_like(frame2_left)

    # Make image saturation to a maximum value
    hsv_mask[..., 1] = 255

    # Set image hue value according to the angle of optical flow
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    
    # Set value as per the normalized magnitude of optical flow
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to rgb
    rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)

    # Resizing the images
    rgb_representation = imutils.resize(rgb_representation, width=800)

    # Showing the pictures
    cv2.imshow('Optical Flow', rgb_representation)
    kk = cv2.waitKey(20) & 0xff
    # Press 'e' to exit the video
    if kk == ord('e'):
        break
    # Press 's' to save the video
    elif kk == ord('s'):
        cv2.imwrite('Optical_imag_lefte.png', following)
        cv2.imwrite('HSV_converted_imag_lefte.png', rgb_representation)
    prvs = following

capture_left.release()
capture_right.release()
cv2.destroyAllWindows()
