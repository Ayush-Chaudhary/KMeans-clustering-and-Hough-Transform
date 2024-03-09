# Given an image, compute and display two histograms of its hue values. Let the
# first histogram use equally-spaced bins (uniformly dividing up the hue values),
# and let the second histogram use bins defined by the k cluster center
# memberships (ie., all pixels belonging to hue cluster i go to the i-th bin, for
# i=1,...k). Use the following form:
# function [histEqual, histclustered]= getHueHists(im, k)
# where im is an MXNX3 matrix represeting an RGB image, and histEqual and
# histClustered are the two output histograms

import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def getHueHists(im, k):
    # Convert the image to HSV
    hsvimg = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    # Reshape the image to be a list of pixels
    pixels = hsvimg.reshape((-1, 3))
    # Compute the histogram of the hue channel with equally spaced k bins
    histEqual = np.histogram(pixels[:, 0], bins=k, range=(0, 180))[0]
    # Fit a k-means estimator to the Hue channel
    estimator = KMeans(n_clusters=k)
    estimator.fit(pixels[:, 0].reshape(-1, 1))
    # Quantize the hue space
    labels = estimator.predict(pixels[:, 0].reshape(-1, 1))
    # Compute the histogram of the hue channel with k bins
    histClustered = np.histogram(labels, bins=k, range=(0, k))[0]

    # plot the histograms
    plt.figure()
    plt.subplot(2,1,1)
    plt.bar(range(k), histEqual)
    plt.title('Equal bins')
    plt.subplot(2,1,2)
    plt.bar(range(k), histClustered)
    plt.title('Clustered bins')
    plt.show()
    
    return histEqual, histClustered
