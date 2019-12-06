import cv2;
import numpy as np;

# Read image
im_in = cv2.imread("acnepaint.png", cv2.IMREAD_GRAYSCALE);

# Threshold.
# Set values equal to or above 220 to 0.
# Set values below 220 to 255.


detector=cv2.SimpleBlobDetector_create()
keypoints= detector.detect(im_in)



#cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensure the
#size of circle corresponds to the size of blob
blank=np.zeros((1,1))
blobs=cv2.drawKeypoints(im_in,keypoints,blank,(0,255,255),cv2.DRAW_MATCHES_FLAGS_DEFAULT)

#th, im_th = cv2.threshold(im_in, 220, 255, cv2.THRESH_BINARY_INV);
th= blobs
im_th = blobs
# Copy the thresholded image.
im_floodfill = im_th.copy()

# Mask used to flood filling.
# Notice the size needs to be 2 pixels than the image.
h, w = im_th.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)

# Floodfill from point (0, 0)
cv2.floodFill(im_floodfill, mask, (0, 0), 255);

# Invert floodfilled image
im_floodfill_inv = cv2.bitwise_not(im_floodfill)

# Combine the two images to get the foreground.
im_out = im_th | im_floodfill_inv

h,w,bpp = np.shape(im_out)
pixels = im_floodfill_inv

# Display images.
cv2.imshow("Blob Detection Image", im_th)
cv2.imshow("Floodfilled Image", im_floodfill)
cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
cv2.imshow("Foreground", im_out)
cv2.waitKey(0)