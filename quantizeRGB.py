# write code to quantize a color space by applying k-means
# clustering to the pixels in a given input image, and experiment with two different color
# spacesRGB and HSV

# Given an RGB image, quantize the 3-dimensional RGB space, and map each
# pixel in the input image to its nearest k-means center. That is, replace the RGB
# value at each pixel with its nearest cluster's average RGB value

# imports
import numpy as np
from sklearn.cluster import KMeans

# Use built in kmeans function from sklearn

def quantizeRGB(origimg, k):
    # Reshape the image to be a list of pixels
    pixels = origimg.reshape((-1, 3))
    # Fit a k-means estimator
    estimator = KMeans(n_clusters=k, n_init=10)
    estimator.fit(pixels)
    # Replace the pixels with their centers
    labels = estimator.predict(pixels)
    centers = estimator.cluster_centers_
    newimg = centers[labels].reshape(origimg.shape)
    # convert image to uint8
    newimg = np.array(newimg, dtype=np.uint8)
    centers = np.array(centers, dtype=np.uint8)
    return newimg, centers