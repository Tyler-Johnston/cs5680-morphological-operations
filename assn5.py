import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import cv2

# PROBLEM 1 PART "A"
wirebondIm = cv2.imread('Wirebond.tif', cv2.IMREAD_GRAYSCALE)

# PROBLEM 1 PART "B"
# mediumRectangleSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
# mediumRectangleOutput = cv2.morphologyEx(wirebondIm, cv2.MORPH_OPEN, mediumRectangleSE)

# Define the structuring elements
dilateSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mediumEllipseSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))

# Erode the image with mediumEllipseSE using cv2.morphologyEx()
eroded_image = cv2.morphologyEx(wirebondIm, cv2.MORPH_ERODE, mediumEllipseSE)

# Then dilate the eroded image using cv2.morphologyEx()
refined_image_b_4 = cv2.morphologyEx(eroded_image, cv2.MORPH_DILATE, dilateSE)

# dilateSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
# # Erode the image with this structuring element
# horizontally_thinned_image = cv2.erode(wirebondIm, mediumRectangleSE, iterations=1)
# # Optionally, dilate the eroded image slightly to restore some of the shape vertically
# refined_image_b_3 = cv2.dilate(horizontally_thinned_image, dilateSE, iterations=1)

# PROBLEM 1 PART "D"
largeRectangleSE = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))
largeRectangleOutput = cv2.morphologyEx(wirebondIm, cv2.MORPH_OPEN, largeRectangleSE)

# plotting
plt.figure(figsize=(10, 5)) # Figure 1
plt.suptitle("Problem 1, Part 1: Wirebond.tif morphological operations")
plt.subplot(1, 4, 1)
plt.imshow(wirebondIm, cmap='gray')
plt.axis("off")
plt.title("A: Original Image")

# plt.subplot(1, 4, 2)
# plt.imshow(refined_image_b_4, cmap='gray')
# plt.axis("off")
# plt.title("B")

plt.subplot(1, 4, 3)
plt.imshow(refined_image_b_4, cmap='gray')
plt.title("C")

plt.subplot(1, 4, 4)
plt.imshow(largeRectangleOutput, cmap='gray')
plt.axis("off")
plt.title("D")
plt.tight_layout()

plt.show()