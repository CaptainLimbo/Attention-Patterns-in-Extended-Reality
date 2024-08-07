from hints import *
import numpy as np

SOLVED_PUZZLE = np.array([
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
])
INPUT_PUZZLE = np.array([
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 0, 0, 0, 5, 6, 7],
    [8, 0, 9, 0, 6, 0, 0, 0, 3],
    [4, 0, 6, 8, 0, 3, 0, 0, 1],
    [7, 0, 3, 0, 2, 0, 8, 0, 6],
    [9, 6, 0, 0, 0, 0, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 0, 0, 0, 8, 6, 1, 7, 9],
])
INPUT_WRONG_PUZZLE = np.array([
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 0, 0, 0, 5, 6, 7],
    [8, 7, 9, 0, 6, 0, 0, 0, 3],
    [4, 0, 6, 8, 0, 3, 0, 0, 1],
    [7, 0, 3, 0, 2, 0, 8, 0, 6],
    [9, 6, 0, 0, 0, 0, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 0, 0, 0, 8, 6, 1, 7, 9],
])


def find_error_and_question_hints(
    input_puzzle,
    solved_puzzle
):
    # first deal with error hints
    for r in range(9):
        for c in range(9):
            if input_puzzle[r][c] == 0 or input_puzzle[r][c] == solved_puzzle[r][c]:
                continue
            digit = input_puzzle[r][c]
            cell = (r, c)
            input_puzzle[r][
                c
            ] = 0  
            if digit in input_puzzle[r]:
                hint = ObservableErrorHint(
                    digit=digit,
                    cell=cell,
                    reference=(r, np.where(input_puzzle[r] == digit)[0][0]),
                    host="row",
                    index=r,
                )
                return hint

            elif digit in input_puzzle[:, c]:

                hint = ObservableErrorHint(
                    digit=digit,
                    cell=cell,
                    reference=(np.where(input_puzzle[:, c] == digit)[0][0], c),
                    host="column",
                    index=c,
                )
                return hint
            else:
                box_y, box_x = r // 3, c // 3
                if (
                    digit
                    in input_puzzle[
                        box_y * 3 : box_y * 3 + 3, box_x * 3 : box_x * 3 + 3
                    ]
                ):
                    rr, rc = np.where(
                        input_puzzle[
                            box_y * 3 : box_y * 3 + 3, box_x * 3 : box_x * 3 + 3
                        ]
                        == digit
                    )
                    hint = ObservableErrorHint(
                        digit=digit,
                        cell=cell,
                        reference=(box_y * 3 + rr[0], box_x * 3 + rc[0]),
                        host="box",
                        index=box_y * 3 + box_x,
                    )
                    return hint
                else:  # it's wrong and there's no obvious conflict
                    return NotObviousErrorHint(digit=digit, cell=cell)
    return None


def find_all_positive_hints(
    input_puzzle,
    solved_puzzle,
):
    last_remaining_hint = find_last_remaining_hints(
        input_puzzle,
        solved_puzzle,
    )
    return (
        last_remaining_hint
        if last_remaining_hint
        else find_last_possible_number_hints(
            input_puzzle,
        )
    )


def find_last_remaining_hints(
    input_puzzle,
    solved_puzzle,
):
    """
    Generate hints of type 1: looking at a box, use digits to scan rows/columns,
    eliminate all possible positions and the only remaining cell is the place for that digit.
    This also holds similarly for row/column based.
    We implement then by the order of box, row and column.
    """
    # first go over the recognized puzzle and find the constraints of each digit
    available_cells = {
        i: set(
            [(y, x) for x in range(9) for y in range(9) if input_puzzle[y][x] == 0]
        )
        for i in range(1, 10)
    }
    last_free_set = set()
    processed = set()
    temp_hints = []

    for y in range(9):
        for x in range(9):
            if input_puzzle[y][x] == 0:
                continue
            digit = input_puzzle[y][x]
            # constraints: cannot be placed in the same row, column or box
            to_remove = set(
                [(y, i) for i in range(9)]
                + [(i, x) for i in range(9)]
                + [(y // 3 * 3 + i, x // 3 * 3 + j) for i in range(3) for j in range(3)]
            )
            available_cells[digit] -= to_remove
    for i in range(3):
        for j in range(3):
            box = set(
                [
                    (i * 3 + y, j * 3 + x)
                    for x in range(3)
                    for y in range(3)
                    if input_puzzle[i * 3 + y][j * 3 + x] == 0
                ]
            )
            # don't proceed if there's only one or no empty cell in the box
            if len(box) == 1:
                cell = box.pop()
                last_free_set.add(cell)
                processed.add(cell)
                last_free_hint = LastFreeHint(
                    host="box", cell=cell, digit=solved_puzzle[cell[0]][cell[1]]
                )
                return last_free_hint
            if len(box) == 0:
                continue
            for digit in range(1, 10):
                intersection = box & available_cells[digit]
                if len(intersection) == 1:
                    cell = intersection.pop()
                    hint = look_for_hints_from_box(
                        input_puzzle,
                        i,
                        j,
                        digit,
                        cell,
                    )
                    if hint:
                        processed.add(cell)
                        temp_hints.append(hint)
    for i in range(9):
        row = set([(i, j) for j in range(9) if input_puzzle[i][j] == 0])
        if len(row) == 1:
            cell = row.pop()
            if (
                cell not in last_free_set
            ):  
                last_free_hint = LastFreeHint(
                    host="row", cell=cell, digit=solved_puzzle[cell[0]][cell[1]]
                )
                last_free_set.add(cell)
                processed.add(cell)
                return last_free_hint
        if len(row) == 0 or temp_hints != []:
            continue
        for digit in range(1, 10):
            intersection = row & available_cells[digit]
            if len(intersection) == 1:
                cell = intersection.pop()
                if cell not in processed:
                    hint = look_for_hints_from_row(
                        input_puzzle,
                        i,
                        digit,
                        cell,
                    )
                    if hint:
                        processed.add(cell)
                        temp_hints.append(hint)
    for j in range(9):
        column = set([(i, j) for i in range(9) if input_puzzle[i][j] == 0])
        if len(column) == 1:
            cell = column.pop()
            if cell not in last_free_set:
                processed.add(cell)
                last_free_set.add(cell)
                last_free_hint = LastFreeHint(
                    host="column", cell=cell, digit=solved_puzzle[cell[0]][cell[1]]
                )
                return last_free_hint
        if len(column) == 0 or temp_hints != []:
            continue
        for digit in range(1, 10):
            intersection = column & available_cells[digit]
            if len(intersection) == 1:
                cell = intersection.pop()
                if cell not in processed:
                    hint = look_for_hints_from_column(
                        input_puzzle,
                        j,
                        digit,
                        cell,
                    )
                    if hint:
                        temp_hints.append(hint)
    # return all_hints
    if temp_hints != []:
        return temp_hints[0]
    return None


def find_last_possible_number_hints(
    input_puzzle,
):
    """
    generate last possible number hints
    """
    # first initialize the available digits for each cell
    available_digits = dict(
        [
            ((i, j), list(range(1, 10)))
            for i in range(9)
            for j in range(9)
            if input_puzzle[i][j] == 0
        ]
    )

    # filter out the digits for each cell
    for y in range(9):
        for x in range(9):
            if input_puzzle[y][x] != 0:
                digit = input_puzzle[y][x]
                to_remove = set(
                    [(y, i) for i in range(9)]
                    + [(i, x) for i in range(9)]
                    + [
                        (y // 3 * 3 + i, x // 3 * 3 + j)
                        for i in range(3)
                        for j in range(3)
                    ]
                )
                for cell in to_remove:
                    if cell in available_digits and digit in available_digits[cell]:
                        available_digits[cell].remove(digit)

    # go over the puzzle again and see where the hint comes from
    for cell, remaining_candidates in available_digits.items():
        if len(remaining_candidates) == 1:
            # search in the order of row, column, box
            y, x = cell
            digit = remaining_candidates[0]
            to_find = set(range(1, 10))
            to_find.remove(digit)
            reference_cells = set()
            rows, columns, boxes = [], [], []
            # start with row
            for j in range(9):
                if input_puzzle[y][j] in to_find:
                    reference_cells.add((y, j))
                    rows = [y]
                    to_find.remove(input_puzzle[y][j])
            # then column
            for i in range(9):
                if to_find == set():
                    break
                if input_puzzle[i][x] in to_find:
                    reference_cells.add((i, x))
                    columns = [x]
                    to_find.remove(input_puzzle[i][x])

            # then box
            box_y, box_x = y // 3, x // 3
            for i in range(3):
                for j in range(3):
                    if to_find == set():
                        break
                    if input_puzzle[box_y * 3 + i][box_x * 3 + j] in to_find:
                        reference_cells.add((box_y * 3 + i, box_x * 3 + j))
                        boxes = [box_y * 3 + box_x]
                        to_find.remove(input_puzzle[box_y * 3 + i][box_x * 3 + j])
            last_possible_number_hint = LastPossibleNumberHint(
                cell=cell,
                digit=digit,
                box_id=boxes,
                row_id=rows,
                column_id=columns,
                digit_positions=reference_cells,
            )
            return last_possible_number_hint
    return None


def look_for_hints_from_box(
    input_puzzle,
    box_i,
    box_j,
    digit,
    cell,
):
    # if digit exists, means that we know that the last_remaining_cell hint for that digit is available
    possible_rows = [box_i * 3 + i for i in range(3) if box_i * 3 + i != cell[0]]
    possible_columns = [box_j * 3 + i for i in range(3) if box_j * 3 + i != cell[1]]
    hint_rows, hint_columns = [], []
    covered, positions = set(), set()
    for row in possible_rows:
        # if the row in the box is full of printed digits, no need to put hint
        if all(
            [input_puzzle[row][i] != 0 for i in range(box_j * 3, box_j * 3 + 3)]
        ):
            continue
        # else look for the digit in the full row
        if digit in input_puzzle[row]:
            pos = np.where(input_puzzle[row] == digit)[0][0]
            hint_rows.append(row)
            covered.update([(row, i) for i in range(box_j * 3, box_j * 3 + 3)])
            positions.add((row, pos))
    for column in possible_columns:
        # if the column in the box is full of printed digits, or if already covered by rows, no need to put hint
        if all(
            [
                input_puzzle[i][column] != 0 or (i, column) in covered
                for i in range(box_i * 3, box_i * 3 + 3)
            ]
        ):
            continue
        # else look for the digit in the full column
        if digit in input_puzzle[:, column]:
            pos = np.where(input_puzzle[:, column] == digit)[0][0]
            hint_columns.append(column)
            positions.add((pos, column))
    if hint_rows == [] and hint_columns == []:
        return None
    return LastRemainingHint(
        cell=cell,
        host="box",
        digit=digit,
        box_id=[box_i * 3 + box_j],
        row_id=hint_rows,
        column_id=hint_columns,
        digit_positions=positions,
    )


def look_for_hints_from_row(
    input_puzzle, row_i, digit, cell
):
    # if digit exists, means that we know that the last_remaining_cell hint for that digit is available
    possible_columns = [
        i for i in range(9) if i != cell[1] and input_puzzle[row_i][i] == 0
    ]
    box_i = cell[0] // 3
    possible_box_j = [j for j in range(3) if j != cell[1] // 3]

    hint_columns, hint_boxes = [], []
    covered, positions = set(), set()

    for box_j in possible_box_j:
        if all(
            [input_puzzle[row_i][x] != 0 for x in range(box_j * 3, box_j * 3 + 3)]
        ):
            continue
        box = input_puzzle[box_i * 3 : box_i * 3 + 3, box_j * 3 : box_j * 3 + 3]
        if digit in box:
            r, c = np.where(box == digit)
            hint_boxes.append(box_i * 3 + box_j)
            covered.update([(row_i, x) for x in range(box_j * 3, box_j * 3 + 3)])
            positions.add((box_i * 3 + r[0], box_j * 3 + c[0]))

    for column in possible_columns:
        if (row_i, column) in covered:
            continue
        if digit in input_puzzle[:, column]:
            pos = np.where(input_puzzle[:, column] == digit)[0][0]
            hint_columns.append(column)
            positions.add((pos, column))

    if hint_boxes == [] and hint_columns == []:
        return None
    return LastRemainingHint(
        cell=cell,
        host="row",
        digit=digit,
        box_id=hint_boxes,
        row_id=[row_i],
        column_id=hint_columns,
        digit_positions=positions,
    )


def look_for_hints_from_column(
    input_puzzle, column_j, digit, cell
):
    # if digit exists, means that we know that the last_remaining_cell hint for that digit is available
    possible_rows = [
        i for i in range(9) if i != cell[0] and input_puzzle[i][column_j] == 0
    ]
    box_j = cell[1] // 3
    possible_box_i = [j for j in range(3) if j != cell[0] // 3]

    hint_rows, hint_boxes = [], []
    covered, positions = set(), set()
    for box_i in possible_box_i:
        # if the column in the box is full of digits, no need to put hint
        if all(
            [
                input_puzzle[y][column_j] != 0
                for y in range(box_i * 3, box_i * 3 + 3)
            ]
        ):
            continue
        box = input_puzzle[box_i * 3 : box_i * 3 + 3, box_j * 3 : box_j * 3 + 3]
        if digit in box:
            r, c = np.where(box == digit)
            hint_boxes.append(box_i * 3 + box_j)
            covered.update([(y, column_j) for y in range(box_i * 3, box_i * 3 + 3)])
            positions.add((box_i * 3 + r[0], box_j * 3 + c[0]))
    for row in possible_rows:
        if (row, column_j) in covered:
            continue
        if digit in input_puzzle[row]:
            pos = np.where(input_puzzle[row] == digit)[0][0]
            hint_rows.append(row)
            positions.add((row, pos))

    if hint_boxes == [] and hint_rows == []:
        return None
    return LastRemainingHint(
        cell=cell,
        host="column",
        digit=digit,
        box_id=hint_boxes,
        row_id=hint_rows,
        column_id=[column_j],
        digit_positions=positions,
    )


if __name__ == "__main__":
    neg_hint = find_error_and_question_hints(
        INPUT_WRONG_PUZZLE,
        SOLVED_PUZZLE
    )
    print("Negative Hint: ")
    print(neg_hint)

    pos_hint = find_all_positive_hints(
        INPUT_PUZZLE,
        SOLVED_PUZZLE
    )
    print("Positive Hint: ")
    print(pos_hint)
