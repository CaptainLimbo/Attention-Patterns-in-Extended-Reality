# USAGE
# python solve_sudoku_puzzle.py --model output/digit_classifier.h5 --image sudoku_puzzle.jpg

# import the necessary packages

# BEGIN: yzq5v8z7j5q1
import sys
import os

# END: yzq5v8z7j5q1
import copy
from pyimagesearch.sudoku import extract_digit
from pyimagesearch.sudoku import find_puzzle
from solver.sudokuSolver import solveSudoku
from solver.dlx_sudoku_solver import solve_dlx

# from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2
import math
from itertools import combinations
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.preprocessing.image import img_to_array
import tensorflow as tf
from tensorflow.keras.models import load_model

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to trained digit classifier")
ap.add_argument(
    "-i", "--image", required=True, help="path to input sudoku puzzle image"
)
ap.add_argument(
    "-d",
    "--debug",
    type=int,
    default=-1,
    help="whether or not we are visualizing each step of the pipeline",
)
args = vars(ap.parse_args())

# load the digit classifier from disk
print("[INFO] loading digit classifier...")
model = tf.saved_model.load(args["model"])

# load the input image from disk and resize it
print("[INFO] processing image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=800)

# find the puzzle in the image and then
(puzzleImage, warped) = find_puzzle(image, debug=args["debug"] > 0)


# def prediction function
def pred_digit(logit):
    sort_args = np.argsort(logit)
    index = len(sort_args) - 1
    max_1 = sort_args[index]
    first = logit[max_1]
    board[y, x] = max_1
    while index > 0:
        index -= 1
        max_2 = sort_args[index]
        second = logit[max_2]
        frac = math.exp(second) / math.exp(first)
        if frac < 0.1:
            break
        print(f"Appending to {(y, x)}: candidate: {max_2}, frac: {frac}")
        second_candidate_list.append([(y, x), (max_2, frac)])


# initialize our 9x9 sudoku board
board = np.zeros((9, 9), dtype="int")

# a sudoku puzzle is a 9x9 grid (81 individual cells), so we can
# infer the location of each cell by dividing the warped image
# into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

# initialize a list to store the (x, y)-coordinates of each cell
# location
cellLocs = []
candidate_list = []
# prepare a second_candidate dict
second_candidate_list = []
# loop over the grid locations
for y in range(0, 9):
    # initialize the current list of cell locations
    row = []
    candidate_list.append([])
    for x in range(0, 9):
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY

        # add the (x, y)-coordinates to our cell locations list
        row.append((startX, startY, endX, endY))

        # crop the cell from the warped transform image and then
        # extract the digit from the cell
        cell = warped[startY:endY, startX:endX]
        digit = extract_digit(cell, debug=args["debug"] > 0)

        # verify that the digit is not empty
        if digit is not None:
            # cell = cv2.resize(cell, (28, 28))
            # digit = cv2.resize(digit, (28, 28))
            foo = np.hstack([cell, digit])
            cv2.imshow("Cell/Digit", foo)


            digit = cv2.resize(digit, (28, 28))
            # normalize to 0-1
            digit = digit.astype("float") / 255.0
            roi = img_to_array(digit)
            roi = np.expand_dims(roi, axis=0)
            pred = model(roi).numpy()

            board[y, x] = pred.argmax(axis=1)[0]
            if args["debug"] > 0:
                print(f"We are getting {pred.argmax(axis=1)[0]} at grid ({y}, {x}).")
        else:
            candidate_list[y].append([-1])
    cellLocs.append(row)

# solve the puzzle
res = solveSudoku(board.tolist())
print("Result: ", res)

print("My board: ", board.tolist())
