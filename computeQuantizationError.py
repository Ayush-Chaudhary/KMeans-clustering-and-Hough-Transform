# Write a function to compute the SSD error (sum of squared error) between the
# original RGB pixel values and the quantized values, with the following form:
# function [error] = computeQuantizationError (origImg,quantizedImg)
# where origImg and quantized Img are both RGB images, and error is a scalar
# giving the total SSD error across the image

import numpy as np

def computeQuantizationError(origImg, quantizedImg):
    # Reshape the image to be a list of pixels
    pixelsOrig = origImg.reshape((-1, 3))
    pixelsQuantized = quantizedImg.reshape((-1, 3))
    # Compute the SSD error
    error = np.sum(np.square(pixelsOrig - pixelsQuantized))#/len(pixelsOrig)
    return error