# Sudoku Server
The python-backend server for the Sudoku AR application. Given an image of the puzzle ([example image](examples/sudoku_example_image.jpg)), can solve the puzzle and generate the hint for the next step. This repo is partly insprired by the post [OpenCV Sudoku Solver and OCR](https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/) by Adrian Rosebrock.

## Sudoku Recognition
Please check [OpenCV Sudoku Solver and OCR](https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/) for detailed descriptions of the Sudoku recognition pipeline. The procedure can be roughly summarized as:
1. extract the largest convex contour with 4 corners
2. apply a four-point perspective transform to obtain a top-down birds-eye view
3. divide the puzzle into 81 equal-sized cells to extract the digits in each cell

## Digit Recognition
Unlike [OpenCV Sudoku Solver and OCR](https://pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/), our digit recognition uses the Efficient-CapsNet model, available [here](https://github.com/EscVM/Efficient-CapsNet). Model checkpoint (trained on MNIST) is provided at [src/digit_classification/checkpoint/saved_model](src/digit_classification/checkpoint/saved_model), but you can also train your own model using [src/digit_classification/train_capsnet.py](src/digit_classification/train_capsnet.py). Parameters we used are provided at [src/digit_classification/flags.txt](src/digit_classification/flags.txt)

## Sudoku Solver
If you are interested, take a look at [src/sudoku_solver/solver](src/sudoku_solver/solver) for algorithms to solve any given puzzle. Here we provide some details on the implementation of a Sudoku solver using the Dancing Links X algorithm.

### Dancing Links X
[dlx_sudoku_solver.py](src/sudoku_solver/solver/dlx_sudoku_solver.py) implements a dancing-link-X based solver, which is supposed to be the fastest. See (this link)[https://rafal.io/posts/solving-sudoku-with-dancing-links.html] for more detailed explanation of the algorithm. 

Dancing Links is an algorithm for exact cover problems, which for a matrix composed of 0 and 1, aims to find a set of rows such that each column in the set has exactly one 1. It does that by deleting the rows that contain a 1 on the same column of the row just found to create a sub-problem. 

**There are generally 4 constraints for a sudoku problem:**
1. Each cell could have only one digit
2. For each row, digits 1-9 have to appear once and only once
3. For each column, digits 1-9 have to appear once and only once
4. For each box, digits 1-9 have to appear once and only once

##### For the first constraint, we define the 1-81 columns of the matrix to be
column 1: (1, 1) has a digit
column 2: (1, 2) has a digit
...
column 81: (9, 9) has a digit

##### For the second constraint, we define the 82-162 columns of the matrix to be
column 82: digit 1 in row 1
column 83: digit 2 in row 1
...
column 162: digit 9 in row 9

##### For the third constraint, we define the 163-243 columns of the matrix to be
column 163: digit 1 in column 1
...
column 243: digit 9 in column 9

##### For the fourth constraint, we define the 244-324 columns of the matrix to be
column 244: digit 1 in box 1
...
column 324: digit 9 in box 9

Then for any given cell, for example a 7 in cell (4, 2), we could convert it into **1 row**, where:
1. A digit in (4, 2) -> column N = (4-1) * 9 + 2 = 29 has a 1
2. A 7 in row 4 -> column N = (4-1)* 9 + 7 + 81 = 115 has a 1
3. A 7 in column 2 -> column N = (2-1)* 9 + 7 + 162 = 178 has a 1
4. A 7 in box 4 -> column N = (4-1)* 9 + 7 + 243 = 277 has a 1

And since column 29 is 1 only at this row, we know there won't be a mistake;

For any cell that doesn't have a digit, we convert to 9 rows, for each digit, in the same way.
Then we can simply run Dancing Links X to solve it!

### Constraint Satisfaction
[sudokuSolver.py](src/sudoku_solver/solver/sudokuSolver.py) is a more traditional constraint-satisfaction solution that uses bit computation. DFS is applied, but any other search algorithm should work. Refer to the comments to see how it works. 

To test the pipeline end-to-end, run
```python
python src/sudoku_solver/extract_and_solve.py -i examples/sudoku_example_image.jpg -m src/digit_classification/checkpoint/saved_model
```

## Sudoku Hint Generator
Please refer to [src/hint_generator](src/hint_generator/) for hint generation. [src/hint_generator/hints.py](src/hint_generator/hints.py) defines  the types of hints we are using right now, and [src/hint_generator/my_hint_generator.py](src/hint_generator/my_hint_generator.py) generates the hints based on a solved puzzle and a currently recognized puzzle. Refer to the comments for more detailed explanation.

To test the hint generation module, run
```python
python src/sudoku_solver/hint_generator/my_hint_generator.py
```
Feel free to change the predefined puzzles on top of that file to test on other puzzles!

