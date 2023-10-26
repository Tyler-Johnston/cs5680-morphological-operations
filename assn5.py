import numpy as np
import matplotlib.pyplot as plt
import cv2

wirebondIm = cv2.imread('Wirebond.tif', cv2.IMREAD_GRAYSCALE)
shapesIm = cv2.imread('Shapes.tif', cv2.IMREAD_GRAYSCALE)
dowelsIm = cv2.imread('Dowels.tif', cv2.IMREAD_GRAYSCALE)

# PROBLEM 1 QUESTION 1B
ellipseErode_SE_B = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
erodedEllipseIm_B = cv2.morphologyEx(wirebondIm, cv2.MORPH_ERODE, ellipseErode_SE_B)

# PROBLEM 1 QUESTION 1C
rectangleErode_SE_C = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
erodedRectagleIm_C = cv2.morphologyEx(wirebondIm, cv2.MORPH_ERODE, rectangleErode_SE_C)

# PROBLEM 1 QUESTION 1D
rectangleErode_SE_D = cv2.getStructuringElement(cv2.MORPH_RECT, (35,35))
erodedRectagleIm_D = cv2.morphologyEx(wirebondIm, cv2.MORPH_ERODE, rectangleErode_SE_D)

# plotting
plt.figure(figsize=(10, 5)) # Figure 1
plt.suptitle("Problem 1, Part 1: Wirebond.tif")
plt.subplot(1, 4, 1)
plt.imshow(wirebondIm, cmap='gray')
plt.axis("off")
plt.title("A")

plt.subplot(1, 4, 2)
plt.imshow(erodedEllipseIm_B, cmap='gray')
plt.axis("off")
plt.title("B")

plt.subplot(1, 4, 3)
plt.imshow(erodedRectagleIm_C, cmap='gray')
plt.axis("off")
plt.title("C")

plt.subplot(1, 4, 4)
plt.imshow(erodedRectagleIm_D, cmap='gray')
plt.axis("off")
plt.title("D")
plt.tight_layout()

# PROBLEM 1 QUESTION 1F
rectanlgeOpenSE_F = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
rectanlgeOpenIm_F = cv2.morphologyEx(shapesIm, cv2.MORPH_OPEN, rectanlgeOpenSE_F)

# PROBLEM 1 QUESTION 1G
rectanlgeCloseSE_G = cv2.getStructuringElement(cv2.MORPH_RECT, (20,20))
rectanlgeCloseIm_G = cv2.morphologyEx(shapesIm, cv2.MORPH_CLOSE, rectanlgeCloseSE_G)

# PROBLEM 1 QUESTION 1H
rectangleDilate_SE_H = cv2.getStructuringElement(cv2.MORPH_RECT, (18,18))
dilatedRectagleIm_H = cv2.morphologyEx(rectanlgeOpenIm_F, cv2.MORPH_DILATE, rectangleDilate_SE_H)

# plotting
plt.figure(figsize=(10, 5)) # Figure 2
plt.suptitle("Problem 1, Part 2: Shapes.tif")
plt.subplot(1, 4, 1)
plt.imshow(shapesIm, cmap='gray')
plt.axis("off")
plt.title("E")

plt.subplot(1, 4, 2)
plt.imshow(rectanlgeOpenIm_F, cmap='gray')
plt.axis("off")
plt.title("F")

plt.subplot(1, 4, 3)
plt.imshow(rectanlgeCloseIm_G, cmap='gray')
plt.axis("off")
plt.title("G")

plt.subplot(1, 4, 4)
plt.imshow(dilatedRectagleIm_H, cmap='gray')
plt.axis("off")
plt.title("H")
plt.tight_layout()

# PROBLEM 1 QUESTION 3

radiusList = [2, 3, 4, 5]
res = [dowelsIm, dowelsIm] # close-open image, open-close image

for radius in radiusList:
    diskSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))

    # get previously worked on open close / close open images
    openCloseIm = res.pop()
    closeOpenIm = res.pop()
    
    # Open-Close operation
    openedImage = cv2.morphologyEx(openCloseIm, cv2.MORPH_OPEN, diskSE)
    openCloseOutput = cv2.morphologyEx(openedImage, cv2.MORPH_CLOSE, diskSE)
    
    # Close-Open operation
    closedImage = cv2.morphologyEx(closeOpenIm, cv2.MORPH_CLOSE, diskSE)
    closeOpenOutput = cv2.morphologyEx(closedImage, cv2.MORPH_OPEN, diskSE)

    res.append(closeOpenOutput)
    res.append(openCloseOutput)


# plotting
plt.figure(figsize=(10, 5)) # Figure 3
plt.suptitle("Problem 1, Part 3a: Open-Close")
plt.subplot(1, 2, 1)
plt.imshow(dowelsIm, cmap='gray')
plt.axis("off")
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(res[1], cmap='gray')
plt.axis("off")
plt.title("Open-Close")
plt.tight_layout()

plt.figure(figsize=(10, 5)) # Figure 4
plt.suptitle("Problem 1, Part 3b: Close-Open")
plt.subplot(1, 2, 1)
plt.imshow(dowelsIm, cmap='gray')
plt.axis("off")
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(res[0], cmap='gray')
plt.axis("off")
plt.title("Close-Open")
plt.tight_layout()

plt.show()