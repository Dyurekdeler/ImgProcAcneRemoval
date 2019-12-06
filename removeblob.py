import cv2
import numpy as np


def remove_small_blob():
    # Read image with OpenCv
    img = cv2.imread('acnepaint.png')
    cv2.imshow("origin", img)

    # Convert image to Gray
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply threshold to make image black and white
    ret, img = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
    cv2.imshow("Black-white", img)

    # Find image contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Maximum blobs size
    threshold_blobs_area = 30

    """"# Loop over all contours and fill draw white color for area smaller than threshold.
    for i in range(1, len(contours)):
        index_level = int(hierarchy[0][i][1])
        if index_level <= i:
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            print('Area = ', area)
            if area <= threshold_blobs_area:
                # Draw white color for small blobs
                cv2.drawContours(img, [cnt], -1, 255, -1, 1)"""

remove_small_blob()
cv2.waitKey()
