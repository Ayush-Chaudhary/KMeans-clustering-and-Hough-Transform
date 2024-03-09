# Implement a Hough Transform circle detector that takes an input image and a fixed
# radius, and returns the centers of any detected circles of about that size. Include a
# function with the following form: [centers] = detectCircles (im, radius, useGradient)
# where im is the input image, radius specifies the size of circle we are looking for, and
# useGradient is a flag that allows the user to optionally exploit the gradient direction
# measured at the edge points

import numpy as np
import cv2
import math

def detectCircles(im, radius, useGradient):
    # Convert the image to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Compute the gradient of the image
    if useGradient:
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
        gradient = np.arctan2(sobely, sobelx)
    else:
        gradient = None
    # Compute the Hough Transform
    h, w = gray.shape
    accum = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            if gray[y, x] > 0:
                for theta in range(0, 360):
                    a = int(x - radius * math.cos(math.radians(theta)))
                    b = int(y - radius * math.sin(math.radians(theta)))
                    if a >= 0 and a < w and b >= 0 and b < h:
                        accum[b, a] += 1
    # Find the centers of the circles
    centers = np.where(accum > 0.6 * np.max(accum))
    centers = np.column_stack((centers[1], centers[0]))
    return centers