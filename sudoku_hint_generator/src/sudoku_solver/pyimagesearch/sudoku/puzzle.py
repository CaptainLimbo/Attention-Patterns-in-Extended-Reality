from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_puzzle(image, debug=False):
    # convert the image to grayscale and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blurred = cv2.GaussianBlur(gray, (3, 3), 1)
    # apply adaptive thresholding and then invert the threshold map
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    thresh = cv2.bitwise_not(thresh)

    # check to see if we are visualizing each step of the image
    # processing pipeline (in this case, thresholding)
    if debug:
        cv2.imshow("Puzzle Thresh", thresh)
        cv2.waitKey(0)

    # find contours in the thresholded image and sort them by size in
    # descending order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # initialize a contour that corresponds to the puzzle outline
    puzzleCnt = None

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)


        # if our approximated contour has four points, then we can
        # assume we have found the outline of the puzzle

        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            aspect_ratio = peri / (4 * np.sqrt(area))
            if aspect_ratio > 0.8 and aspect_ratio < 1.2:
                puzzleCnt = approx
                break

    # if the puzzle contour is empty then our script could not find
    # the outline of the sudoku puzzle so raise an error
    if puzzleCnt is None:
        return False, None

    # check to see if we are visualizing the outline of the detected
    # sudoku puzzle
    if debug:
        # draw the contour of the puzzle on the image and then display
        # it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0, 255, 0), 2)
        cv2.imshow("Puzzle Outline", output)
        cv2.waitKey(0)

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down birds eye view
    # of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))

    # check to see if we are visualizing the perspective transform
    if debug:
        # show the output warped image (again, for debugging purposes)
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)

    # return a 2-tuple of puzzle in both RGB and grayscale
    return (True, warped)




def check_center_region(cell, region_percent=0.6):
    # get the dimensions of the cell
    h, w = cell.shape[:2]
    region_h = int(h * region_percent)
    region_w = int(w * region_percent)

    # define the center region
    center_h = h // 2 - region_h // 2
    center_w = w // 2 - region_w // 2
    center_region = cell[center_h : center_h + region_h, center_w : center_w + region_w]

    # check if there are any non-zero pixels in the center region
    return cv2.countNonZero(center_region) > 10


def extract_digit(
    cell, debug=False
):
    # apply automatic thresholding to the cell and then clear any
    if debug:
        cv2.imshow("Original Cell", cell)
        cv2.waitKey(0)
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    if not check_center_region(thresh, 0.7):
        return None
    thresh = clear_border(thresh)

    # check to see if we are visualizing the cell thresholding step
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)

    # find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None


    # otherwise, find the largest contour in the cell and create a mask  for the contour
    # note that for a chinese character, cannot always do so
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, cnts, -1, 255, -1)

    # compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    # if less than 1% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    bound = 0.01
    if percentFilled < bound or percentFilled > 0.4:
        return None

    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    # check to see if we should visualize the masking step
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)

    # return the digit to the calling function
    return digit
