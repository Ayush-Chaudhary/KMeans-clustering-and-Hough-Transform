# Given an RGB image, convert to HSV, and quantize the 1-dimensional Hue
# space. Map each pixel in the input image to its nearest quantized Hue value,
# while keeping its Saturation and Value channels the same as the input. Convert
# the quantized output back to RGB color space. Use the following form:
# [outputImg, meanнues] = quantizeHSV (origImg, k)

import numpy as np
from sklearn.cluster import KMeans
import cv2

def quantizeHSV(origimg, k):
    # Convert the image to HSV
    hsvimg = cv2.cvtColor(origimg, cv2.COLOR_BGR2HSV)
    # Reshape the image to be a list of pixels
    pixels = hsvimg.reshape((-1, 3))
    # Fit a k-means estimator to the Hue channel
    estimator = KMeans(n_clusters=k, n_init=10)
    estimator.fit(pixels[:, 0].reshape(-1, 1))
    # Quantize the hue space
    labels = estimator.predict(pixels[:, 0].reshape(-1, 1))
    centers = estimator.cluster_centers_
    # Replace the hue values with quantized values
    pixels[:, 0] = centers[labels].flatten()
    # Convert the image back to RGB
    newimg = cv2.cvtColor(pixels.reshape(origimg.shape), cv2.COLOR_HSV2BGR)
    # convert image to uint8
    newimg = np.array(newimg, dtype=np.uint8)
    centers = np.array(centers, dtype=np.uint8)
    return newimg, centers
