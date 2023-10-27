import numpy as np
import matplotlib.pyplot as plt
import cv2

wirebondIm = cv2.imread('Wirebond.tif', cv2.IMREAD_GRAYSCALE)
shapesIm = cv2.imread('Shapes.tif', cv2.IMREAD_GRAYSCALE)
dowelsIm = cv2.imread('Dowels.tif', cv2.IMREAD_GRAYSCALE)
smallSquaresIm = cv2.imread('SmallSquares.tif', cv2.IMREAD_GRAYSCALE)
ballIm = cv2.imread('Ball.tif', cv2.IMREAD_GRAYSCALE)

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
plt.tight_layout() #wavelet & fourier transformation has coding questions

# PROBLEM 1 QUESTION 2A - compare "open close" and "close open" with a radius of 5
radius = 5
diskSE = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))

# Open-Close operation
openDowelsIm = cv2.morphologyEx(dowelsIm, cv2.MORPH_OPEN, diskSE)
openCloseOutput = cv2.morphologyEx(openDowelsIm, cv2.MORPH_CLOSE, diskSE)

# Close-Open operation
closeDowelsIm = cv2.morphologyEx(dowelsIm, cv2.MORPH_CLOSE, diskSE)
closeOpenOutput = cv2.morphologyEx(closeDowelsIm, cv2.MORPH_OPEN, diskSE)

# plotting
plt.figure(figsize=(10, 5)) # Figure 3
plt.suptitle("Problem 1, Part 2a: Single Operation")
plt.subplot(1, 2, 1)
plt.imshow(openCloseOutput, cmap='gray')
plt.axis("off")
plt.title("Open-Close")

plt.subplot(1, 2, 2)
plt.imshow(closeOpenOutput, cmap='gray')
plt.axis("off")
plt.title("Close-Open")
plt.tight_layout()

# PROBLEM 1 QUESTION 2B - compare "open close" and "close open" on radius 2, 3, 4, and 5
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
plt.figure(figsize=(10, 5)) # Figure 4
plt.suptitle("Problem 1, Part 2b: Series of Operations")
plt.subplot(1, 2, 1)
plt.imshow(res[1], cmap='gray')
plt.axis("off")
plt.title("Open-Close")

plt.subplot(1, 2, 2)
plt.imshow(res[0], cmap='gray')
plt.axis("off")
plt.title("Close-Open")
plt.tight_layout()

# PROBLEM 1 QUESTION 3

structureElement = np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=np.uint8)
smallSquaresOutput = cv2.morphologyEx(smallSquaresIm, cv2.MORPH_HITMISS, structureElement)

# cv2.MORPH_HITMISS : 1 = foreground, 0 = background, -1 = don't care
foregroundPixelCount = np.sum(smallSquaresIm > 0)
print("number of foreground pixels: ", foregroundPixelCount)

plt.figure(figsize=(10, 5)) # Figure 4
plt.suptitle("Problem 1, Part 3: Foreground Pixels")
plt.subplot(1, 2, 1)
plt.imshow(smallSquaresIm, cmap='gray')
plt.axis("off")
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(smallSquaresOutput, cmap='gray')
plt.axis("off")
plt.title("Result")
plt.tight_layout()

# PROBLEM 2 QUESTION 1

def extract_connected_components_intersection(im, B):
    """Extract connected components from a binary image using intersection in the morphological method."""
    height, width = im.shape
    label_counter = 1
    labeled_image = np.zeros_like(im)
    
    # Convert binary image to boolean for easier operations
    im_bool = im == 255
    
    for i in range(height):
        for j in range(width):
            if im_bool[i, j]:  # If pixel is foreground
                x0 = np.zeros((height, width), dtype=bool)
                x0[i, j] = True
                
                while True:
                    x1 = np.logical_and(cv2.dilate(x0.astype(np.uint8), B), im_bool)
                    if np.array_equal(x0, x1):
                        break
                    x0 = x1
                
                # Label the connected component and remove it from the image
                labeled_image[x1] = label_counter
                im_bool = np.logical_and(im_bool, np.logical_not(x1))
                
                label_counter += 1
                
    return labeled_image

# Extract connected components using intersection
labeled_image_intersection = extract_connected_components_intersection(ballIm, B_large)

# Display the labeled image
plt.figure(figsize=(8, 8))
plt.imshow(labeled_image_intersection, cmap='nipy_spectral')
plt.axis("off")
plt.title("Labeled Image (Intersection Method)")
plt.show()

num_components_intersection = len(np.unique(labeled_image_intersection)) - 1  # Subtract 1 for background
num_components_intersection


# def FindComponentLabels(im, SE):
#     # Initialize the labeled image with zeros (same size as the input image)
#     labelIm = np.zeros_like(im, dtype=np.int32)
    
#     # Initialize label counter
#     label_counter = 1
    
#     # Dictionary to store label equivalences
#     equivalences = {}

#     # First pass
#     for i in range(1, im.shape[0]-1):
#         for j in range(1, im.shape[1]-1):
#             if im[i, j] == 255:  # If pixel is a foreground pixel
#                 neighbors = im[i-1:i+2, j-1:j+2] * SE  # Extract neighbors using the structuring element
#                 labeled_neighbors = labelIm[i-1:i+2, j-1:j+2]
                
#                 # If no labeled neighbors, assign a new label
#                 if np.max(labeled_neighbors) == 0:
#                     labelIm[i, j] = label_counter
#                     label_counter += 1
#                 else:
#                     # Assign the smallest label among the neighbors
#                     min_label = np.min(labeled_neighbors[labeled_neighbors > 0])
#                     labelIm[i, j] = min_label
                    
#                     # Record label equivalences
#                     for label in labeled_neighbors[labeled_neighbors > 0]:
#                         if label != min_label:
#                             equivalences[label] = min_label

#     # Second pass: Resolve label equivalences
#     for i in range(im.shape[0]):
#         for j in range(im.shape[1]):
#             if labelIm[i, j] > 0:
#                 while labelIm[i, j] in equivalences:
#                     labelIm[i, j] = equivalences[labelIm[i, j]]
                
#     # Number of unique components
#     num = len(np.unique(labelIm)) - 1  # Subtract 1 for the background label (0)
    
#     return labelIm, num

# # Define the 3x3 square structuring element for 8-connectivity
# se = np.ones((3, 3), np.uint8)

# # Call the FindComponentLabels function
# labeled_image, num_components = FindComponentLabels(ballIm, se)

# # Display the labeled image
# plt.figure(figsize=(8, 8))
# plt.imshow(labeled_image, cmap='nipy_spectral')
# plt.axis("off")
# plt.title("Labeled Image")
# # plt.show()
# print("number of connected by my own function: ", num_components - 1) # must subtract by 1 because it originally includes the background









# PROBLEM 2 QUESTION 2
builtinNumberOfLabels, builtinLabeledIm = cv2.connectedComponents(ballIm)

plt.figure(figsize=(10, 5)) 
plt.suptitle("Problem 2")
plt.subplot(1, 1, 1)
plt.imshow(builtinLabeledIm, cmap='jet')
plt.axis("off")
plt.title("Original")

print("number of connected by built-in function: ", builtinNumberOfLabels - 1) # must subtract by 1 because it originally includes the background

plt.show()