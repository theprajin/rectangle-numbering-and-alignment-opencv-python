import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv.imread("rectangle.png")

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray, 50, 150)

# Find contours in the edge-detected image with hierarchical retrieval
contours, hierarchy = cv.findContours(edges, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw the contours on
contour_image = image.copy()

# Lists to store the lengths of inner contours (lines)
inner_contour_lengths = []

# Loop through the contours and hierarchy
for i, contour in enumerate(contours):
    # Get the parent index from the hierarchy
    parent_idx = hierarchy[0][i][3]

    if parent_idx != -1:
        # Inner contour, line
        perimeter = cv.arcLength(contour, True)
        inner_contour_lengths.append((perimeter, i))  # Store length and index

# Sort the inner contour lengths in ascending order
inner_contour_lengths.sort()

# Assign numbers to the lines based on their lengths
line_numbers = {
    length_index[1]: i + 1 for i, length_index in enumerate(inner_contour_lengths)
}

# Draw and label the lines for the four contours with lowest lengths
for length, index in inner_contour_lengths[:4]:  # Only the first four contours
    contour = contours[index]
    cv.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)  # Green color
    number = line_numbers[index]
    cv.putText(
        contour_image,
        str(number),
        tuple(contour[0][0]),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )  # Red color

plt.imshow(cv.cvtColor(contour_image, cv.COLOR_BGR2RGB))
plt.show()
