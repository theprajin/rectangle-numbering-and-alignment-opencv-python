import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# Load the image
image = cv.imread("rectangle.png")


# Convert image to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# Apply Gaussian blur to reduce noise
blurred = cv.GaussianBlur(gray, (5, 5), 0)


# Detect edges using Canny
edges = cv.Canny(blurred, 50, 150)


# Find contours
contours, _ = cv.findContours(edges.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)


# Iterate through each contour
for contour in contours:
    # Approximate the contour
    epsilon = 0.05 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)

    # If the contour has four corners, it's likely a rectangle
    if len(approx) == 4:
        # Get the four corners of the rectangle
        pts = np.float32([approx[0], approx[1], approx[2], approx[3]])

        # Define the desired width and height of the rectangle
        width = 200
        height = 300

        # Define the target points for perspective transformation
        target_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])

        # Calculate the perspective transformation matrix
        matrix = cv.getPerspectiveTransform(pts, target_pts)

        # Apply the perspective transformation
        result = cv.warpPerspective(image, matrix, (width, height))

        # Display the result
        cv.imshow("Result", result)
        # plt.imshow(result)
        # plt.show()
        cv.waitKey(0)


# Close all windows
cv.destroyAllWindows()
