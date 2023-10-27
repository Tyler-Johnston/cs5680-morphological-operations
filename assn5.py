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

plt.figure(figsize=(10, 5)) # Figure 5
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

def FindComponentLabels(im, se):
    height, width = im.shape
    numberOfLabels = 0
    labelIm = np.zeros_like(im)
    
    # convert binary image to boolean array for easier operations
    boolIm = im == 255
    
    for i in range(height):
        for j in range(width):
            # check if the current pixel is part of a connected componenet 
            if boolIm[i, j]:
                # create a mask such that the only true value is the current pixel
                x0 = np.zeros((height, width), dtype=bool)
                x0[i, j] = True
                
                # utilize a loop to grow the connected components
                while True:
                    # dilate the current mask using SE and intersect it with boolIm to get a new mask
                    x1 = np.logical_and(cv2.dilate(x0.astype(np.uint8), se), boolIm)

                    # if x0 == x1 then the connected component has reached its max size so stop looping
                    if np.array_equal(x0, x1):
                        break

                    # update x0 to be x1
                    x0 = x1
                
                # Label the connected component and remove it from the image
                labelIm[x1] = numberOfLabels
                boolIm = np.logical_and(boolIm, np.logical_not(x1))
                
                numberOfLabels += 1
                
    return numberOfLabels, labelIm

structureElement2 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
numLabels, labeledIm = FindComponentLabels(ballIm, structureElement2)

print("my own num of labels: ", numLabels)

plt.figure(figsize=(8, 8)) # Figure 6
plt.suptitle("Problem 2, Part 1")
plt.imshow(labeledIm, cmap='gray')
plt.axis("off")
plt.title("Connected Components via FindComponentLabels")

# PROBLEM 2 QUESTION 2
builtinNumberOfLabels, builtinLabeledIm = cv2.connectedComponents(ballIm)

print("number of connected by built-in function: ", builtinNumberOfLabels - 1) # must subtract by 1 because it originally includes the background

plt.figure(figsize=(8, 8)) # Figure 7
plt.suptitle("Problem 2, Part 2")
plt.subplot(1, 1, 1)
plt.imshow(builtinLabeledIm, cmap='gray')
plt.axis("off")
plt.title("Connected Components via Built-In")

# PROBLEM 2 QUESTION 3

def FindBorderConnectedComponents(labelIm):
    height, width = labelIm.shape
    A = np.zeros_like(labelIm)
    borderConnectedComponents = set()

    # iterate through the border of the image and get the labels
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                label = labelIm[i, j]
                if label > 0:
                    borderConnectedComponents.add(label)

    # create the image A with only border-connected components
    for i in range(height):
        for j in range(width):
            label = labelIm[i, j]
            if label in borderConnectedComponents:
                A[i, j] = label

    return len(borderConnectedComponents), A, borderConnectedComponents

numberBorderComponents, A, borderComponents = FindBorderConnectedComponents(labeledIm)
print("number of border components: ", numberBorderComponents)

# plotting
plt.figure(figsize=(10, 5)) # Figure 8
plt.suptitle("t")
plt.subplot(1, 2, 1)
plt.imshow(labeledIm, cmap='gray')
plt.axis("off")
plt.title("Original")

plt.subplot(1, 2, 2)
plt.imshow(A, cmap='gray')
plt.axis("off")
plt.title("Result")
plt.tight_layout()

# PROBLEM 3 QUESTION 4

def getNonBorderIm(labelIm, borderComponents):
    height, width = labelIm.shape
    nonBorderComponents = set()

    for i in range(height):
        for j in range(width):
            label = labelIm[i, j]
            if label > 0 and label not in borderComponents:
                nonBorderComponents.add(label)

    nonBorderIm = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            label = labelIm[i, j]
            if label in nonBorderComponents:
                nonBorderIm[i,j] = label
    
    return nonBorderIm


def FindNonBorderOverlappingComponents(labelIm):
    
    nonBorderIm = getNonBorderIm(labelIm, borderComponents)
    nonBorderIm = (nonBorderIm > 0).astype(np.uint8) * 255
    # Estimate the size of a single particle
    _, non_border_labels = cv2.connectedComponents(nonBorderIm)
    unique_labels, counts = np.unique(non_border_labels, return_counts=True)
    unique_labels = unique_labels[1:]  # Excluding the background label
    counts = counts[1:]
    single_particle_size = np.min(counts)

    # Create a mask for overlapping particles
    overlapping_particles = np.zeros_like(nonBorderIm)
    threshold = 0.2 * single_particle_size

    for label, count in zip(unique_labels, counts):
        if count > single_particle_size + threshold:
            overlapping_particles[non_border_labels == label] = 255
    return overlapping_particles

overlapping_particles = FindNonBorderOverlappingComponents(labeledIm)
    

# FindNonBorderOverlappingComponents(labeledIm, borderComponents)
    

# #  Find connected components using built-in function
# num_labels, labeled_image = cv2.connectedComponents(ballIm)

# # Step 1 & 2: Find all connected components and filter out those that touch the border
# height, width = labeled_image.shape
# border_labels = set()
# for i in range(width):
#     border_labels.add(labeled_image[0, i])
#     border_labels.add(labeled_image[height-1, i])
# for j in range(height):
#     border_labels.add(labeled_image[j, 0])
#     border_labels.add(labeled_image[j, width-1])

# non_border_components = np.copy(labeled_image)
# for label in border_labels:
#     non_border_components[labeled_image == label] = 0

# print(non_border_components)

# # Convert non-border components back to binary for further processing
# binary_non_border = (non_border_components > 0).astype(np.uint8) * 255

# # Estimate the size of a single particle
# _, non_border_labels = cv2.connectedComponents(binary_non_border)
# unique_labels, counts = np.unique(non_border_labels, return_counts=True)
# unique_labels = unique_labels[1:]  # Excluding the background label
# counts = counts[1:]
# single_particle_size = np.min(counts)

# # Create a mask for overlapping particles
# overlapping_particles = np.zeros_like(binary_non_border)
# threshold = 0.2 * single_particle_size

# for label, count in zip(unique_labels, counts):
#     if count > single_particle_size + threshold:
#         overlapping_particles[non_border_labels == label] = 255

# Display the overlapping particles
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(ballIm, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(overlapping_particles, cmap='gray')
plt.title('Overlapping Particles')
plt.axis('off')

plt.tight_layout()
plt.show()


# plt.show()