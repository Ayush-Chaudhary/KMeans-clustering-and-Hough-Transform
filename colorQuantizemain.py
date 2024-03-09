# Write a script colorQuantizemain.py that calls all the above functions
# appropriately using the provided image baloons.jpg, and displays the results.
# Calculate the SSD error for the image quantized in both RGB and HSV space.
# Write down the SSD errors in your answer sheet. Illustrate the quantization with a
# lower and higher value of k

import numpy as np
import cv2
from sklearn.cluster import KMeans
from quantizeRGB import quantizeRGB
from quantizeHSV import quantizeHSV
from computeQuantizationError import computeQuantizationError
from getHueHists import getHueHists
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread('baloons.jpg')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Quantize the image in RGB space
k = 3
newimgRGB, centersRGB = quantizeRGB(img, k)
# Calculate the SSD error
errorRGB = computeQuantizationError(img, newimgRGB)
print('SSD error for RGB quantization with k =', k, 'is', errorRGB)

# Quantize the image in HSV space
newimgHSV, centersHSV = quantizeHSV(img, k)
# Calculate the SSD error
errorHSV = computeQuantizationError(img, newimgHSV)
print('SSD error for HSV quantization with k =', k, 'is', errorHSV)

# Display the results
plt.figure()
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Original image')
plt.subplot(2,2,2)
plt.imshow(cv2.cvtColor(newimgRGB, cv2.COLOR_BGR2RGB))
plt.title('RGB quantization')
plt.subplot(2,2,3)
plt.imshow(cv2.cvtColor(newimgHSV, cv2.COLOR_BGR2RGB))
plt.title('HSV quantization')
plt.show()