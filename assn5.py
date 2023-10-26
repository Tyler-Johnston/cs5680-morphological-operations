import numpy as np
import matplotlib.pyplot as plt
import cv2

# PROBLEM 1 PART "A"
wirebondIm = cv2.imread('Wirebond.tif', cv2.IMREAD_GRAYSCALE)

# PROBLEM 1 PART "C"
rectangleErode_SE_C = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
erodedRectagleIm_C = cv2.morphologyEx(wirebondIm, cv2.MORPH_ERODE, rectangleErode_SE_C)

# PROBLEM 1 PART "D"
rectangleErode_SE_D = cv2.getStructuringElement(cv2.MORPH_RECT, (42,42))
# rectangleDilate_SE_D = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
erodedRectagleIm_D = cv2.morphologyEx(wirebondIm, cv2.MORPH_ERODE, rectangleErode_SE_D)
# dilatedRectagleIm_D = cv2.morphologyEx(erodedRectagleIm_D, cv2.MORPH_DILATE, rectangleDilate_SE_D)

# NW
# Calculate the total amount of shrinkage caused by erosion in both x and y directions
shrinkageX, shrinkageY = rectangleErode_SE_D.shape
rectangleDilate_SE_D = cv2.getStructuringElement(cv2.MORPH_RECT, (shrinkageX, shrinkageY))
dilatedRectagleIm_D = cv2.morphologyEx(erodedRectagleIm_D, cv2.MORPH_DILATE, rectangleErode_SE_D)

# plotting
plt.figure(figsize=(10, 5)) # Figure 1
plt.suptitle("Problem 1, Part 1: Wirebond.tif morphological operations")
plt.subplot(1, 4, 1)
plt.imshow(wirebondIm, cmap='gray')
plt.axis("off")
plt.title("A")

# plt.subplot(1, 4, 2)
# plt.imshow(refined_image_b_4, cmap='gray')
# plt.axis("off")
# plt.title("B")

plt.subplot(1, 4, 3)
plt.imshow(erodedRectagleIm_C, cmap='gray')
plt.title("C")

plt.subplot(1, 4, 4)
plt.imshow(dilatedRectagleIm_D, cmap='gray')
plt.axis("off")
plt.title("D")
plt.tight_layout()

plt.show()