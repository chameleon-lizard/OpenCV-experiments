from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

# Getting the args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", type=str, required=True,
	help="path to input directory of images to stitch")
ap.add_argument("-o", "--output", type=str, required=True,
	help="path to the output image")
ap.add_argument("-m", "--method", type=str, required=True,
	help="method of denoising")
args = vars(ap.parse_args())

# Getting the method of denoising from arguments
method = args["method"]

# Grab the paths to the input images and initialize our images list
print("[INFO] loading images...")
imagePaths = sorted(list(paths.list_images(args["images"])))

# Using list comprehension to add our images to stich list
images = [cv2.imread(imagePath) for imagePath in imagePaths]

# Initializing OpenCV's image sticher object and then performing the image stitching
print("[INFO] stitching images...")
stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
(status, stitched) = stitcher.stitch(images)

# If the status is '0', then OpenCV successfully performed image stitching
if status == 0:
	# Using Bilateral Filter we are getting the least noise (and the least texture), however
	# it is insignificant, because my PC has not enough memory to handle high quality images, so
	# inputs have low detail anyway.
	if method == "Bilateral":
		print("[INFO] denoising images...")
		stitched = cv2.bilateralFilter(stitched, 9, 50, 50)
	# Using Gaussian Filter we are getting very blurry image (well, of course, it's blur after all)
	elif method == "Gaussian":
		print("[INFO] denoising images...")
		stitched = cv2.GaussianBlur(stitched, (3, 3), cv2.BORDER_DEFAULT)
	# Using OpenCV's denoiser we are are getting mixed results. It is using Non-Local Means Denoiser.
	elif method == "FastNl":
		print("[INFO] denoising images...")
		stitched = cv2.fastNlMeansDenoisingColored(stitched, hColor=30)
	elif method == "none":
		pass

	# write the output stitched image to disk
	cv2.imwrite(args["output"], stitched)
	print("[RESULT] successfully stitched images")

	# display the output stitched image to our screen
	cv2.imshow("Stitched", stitched)
	cv2.waitKey(0)

# otherwise the stitching failed, likely due to not enough keypoints) being detected
else:
	if status == 1:
		print("[RESULT] image stitching failed (not enough images)")
	elif status == 2:
		print("[RESULT] image stitching failed (failed to estimate homography matrix)")
	elif status == 3:
		print("[RESULT] image stitching failed (failed to adjust camera parameters)")
