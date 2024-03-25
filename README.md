This is a repository containing my implementation of Hough Transform to detect circles in an image and K-Means clustering using RGB and HSV images to quantize an image values.
The Detailed description to functions and implementation can be found in the pdf file.
The model results can be viewed in the results folder.

Description of Hough Transform Implementation:

i. he detectCircles function takes an input image, the radius of the circles to detect, and optional parameters like the bin threshold, useGradient flag, and post-processing flag.

ii. The input image is converted to grayscale and then smoothed using Gaussian blur to reduce noise.

iii. Canny edge detection is applied to the smoothed image to detect edges.

iv. Parameters for the Hough transform, such as theta range, cosine, and sine values, are computed.

v. Circle candidates are generated based on the given radius and theta values.

vi. A defaultdict named accumulator is created to accumulate votes for circle centers.

vii. If useGradient is True, the gradient direction is calculated using the Sobel operator. Otherwise, the original circle candidates are used.

viii. For each edge pixel in the edge image, the circle candidates are adjusted based on gradient direction (if useGradient is True) and then voted for in the accumulator.

ix. After voting, circles with votes exceeding the bin threshold are shortlisted.

x. Post-processing is optionally applied to remove duplicate circles that are too close to each other.

xi. Finally, the shortlisted circles are drawn on the output image, and the detected circles along with output image(optionally) are returned
