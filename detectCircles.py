import cv2
import numpy as np
from collections import defaultdict

# Use gradient direction to adjust the circle candidates
def calculate_gradient_direction(edge_image):
    gradient_x = cv2.Sobel(edge_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(edge_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_direction = np.arctan2(gradient_y, gradient_x)
    gradient_direction = np.rad2deg(gradient_direction)
    gradient_direction = (gradient_direction + 360) % 360  # Ensure angle is within 0 to 360 degrees
    return gradient_direction

def detectCircles(image, radius, bin_threshold = 0.7, post_process = True, num_thetas= 100, useGradient=False, bin_size = 1):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 1)

    # Apply Canny edge detection
    edge_image = cv2.Canny(gray_image, 100, 200)

    #image size
    img_height, img_width = edge_image.shape[:2]
    
    # R and Theta ranges
    dtheta = int(360 / num_thetas)
    
    # Thetas is bins created from 0 to 360 degree with increment of the dtheta
    thetas = np.arange(0, 360, step=dtheta)
    
    # Calculate Cos(theta) and Sin(theta) it will be required later
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    
    # Evaluate and keep ready the candidate circles dx and dy for different delta radius
    # based on the the parametric equation of circle.
    circle_candidates = []
    for t in range(num_thetas):
        circle_candidates.append((int(radius * cos_thetas[t]), int(radius * sin_thetas[t])))
  
    accumulator = defaultdict(int)

    # calculate gradient direction using edge_image
    gradient_direction = calculate_gradient_direction(edge_image)
    for y in range(img_height):
        for x in range(img_width):
            if edge_image[y][x] != 0: #white pixel
                # Found an edge pixel so now find and vote for circle from the candidate circles passing through this pixel.
                if useGradient:
                    # Use gradient direction to adjust the circle candidates
                    for dx, dy in circle_candidates:
                        adjusted_dx = int(dx * np.cos(np.deg2rad(gradient_direction[y][x]))) + int(dy * np.sin(np.deg2rad(gradient_direction[y][x])))
                        adjusted_dy = int(dx * np.sin(np.deg2rad(gradient_direction[y][x]))) - int(dy * np.cos(np.deg2rad(gradient_direction[y][x])))
                        x_center = x - adjusted_dx
                        y_center = y - adjusted_dy
                        quantized_x = int(x_center / bin_size) * bin_size
                        quantized_y = int(y_center / bin_size) * bin_size
                        accumulator[(quantized_x, quantized_y)] += 1  # vote for current candidate
                else:
                    for rcos_t, rsin_t in circle_candidates:
                        x_center = x - rcos_t
                        y_center = y - rsin_t
                        quantized_x = int(x_center / bin_size) * bin_size
                        quantized_y = int(y_center / bin_size) * bin_size
                        accumulator[(quantized_x, quantized_y)] += 1 #vote for current candidate
  
    # Output image with detected lines drawn
    output_img = image.copy()
    # Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold) 
    out_circles = []
  
    # Sort the accumulator based on the votes for the candidate circles 
    for candidate_circle, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
        x, y = candidate_circle
        current_vote_percentage = votes / num_thetas
        if current_vote_percentage >= bin_threshold: 
            # Shortlist the circle for final result
            out_circles.append((x, y))
      
  
    # Post process the results, can add more post processing later.
    if post_process :
        pixel_threshold = 5
        postprocess_circles = []
        for x, y in out_circles:
            # Exclude circles that are too close of each other
            # all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc, v in postprocess_circles)
            # Remove nearby duplicate circles based on pixel_threshold
            if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold for xc, yc in postprocess_circles):
                postprocess_circles.append((x, y))
        out_circles = postprocess_circles  
    
    # Draw shortlisted circles on the output image
    for x, y in out_circles:
        output_img = cv2.circle(output_img, (x,y), radius, (0,255,0), 2)
  
    return out_circles