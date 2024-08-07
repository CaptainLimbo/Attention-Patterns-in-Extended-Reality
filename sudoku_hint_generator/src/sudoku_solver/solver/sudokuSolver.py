import numpy as np


def solveSudoku(my_board):
    board = my_board.copy()

    def flip(i, j, digit):
        # constraints at (i, j)
        line[i] ^= 1 << digit
        column[j] ^= 1 << digit
        block[i // 3][j // 3] ^= 1 << digit

    def dfs(pos):
        nonlocal valid
        if pos == len(spaces):
            valid = True
            return True

        # forward
        i, j = spaces[pos]
        mask = ~(line[i] | column[j] | block[i // 3][j // 3]) & 0x1FF
        while mask:
            digitMask = mask & (-mask)
            digit = bin(digitMask).count("0") - 1
            flip(i, j, digit)
            board[i][j] = digit + 1
            dfs(pos + 1)
            flip(i, j, digit)
            mask &= mask - 1
            if valid:
                return True

        return False  # not found means we have a wrong puzzle

    line = [0] * 9
    column = [0] * 9
    block = [[0] * 3 for _ in range(3)]
    valid = False
    spaces = list()

    # check
    # bit computation cannot find mistakes in the given puzzle
    bd = np.array(board)
    if any(
        len(np.unique(row)) + (list(row).count(0) - 1 if 0 in list(row) else 0)
        != len(row)
        for row in bd
    ) or any(
        len(np.unique(col)) + (list(col).count(0) - 1 if 0 in list(col) else 0)
        != len(col)
        for col in bd.T
    ):
        return False

    # initialize
    for i in range(9):
        for j in range(9):
            if board[i][j] != 0:
                digit = board[i][j] - 1
                flip(i, j, digit)

    # first pick the cells that's doable
    while True:
        modified = False
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    mask = ~(line[i] | column[j] | block[i // 3][j // 3]) & 0x1FF
                    if not (mask & (mask - 1)):
                        digit = bin(mask).count("0") - 1
                        flip(i, j, digit)
                        board[i][j] = digit + 1
                        modified = True
        if not modified:
            break

    # those that wait to be filled
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                spaces.append((i, j))

    return False if not dfs(0) else board
