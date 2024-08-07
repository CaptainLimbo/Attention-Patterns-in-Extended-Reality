class Hint:
    def __init__(self) -> None:
        """
        priority: the priority of the hint, the smaller the more important. Negative hints typically have negative priority.
        negative: whether the hint is a negative hint, i.e. whether the hint is for a cell that the user filled in a wrong digit.
        HOST_MAP: A dictionary that maps the defined host of the hint to an int, so that it could be sent to the client. Not all hints would have a host.
        """
        self.priority = 0
        self.negative = False
        self.HOST_MAP = {"box": 0, "row": 1, "column": 2}



class LastRemainingHint(Hint):
    def __init__(
        self,
        cell=(0, 0),
        host="box",
        digit=1,
        box_id=[],
        row_id=[],
        column_id=[],
        digit_positions=set(),
    ):
        """
        host: either box, row or column, the place where last remaining happens;
        digit: the digit to be filled in the last remaining cell;
        box(row, column)_id: the box(row, column) (either as host or eliminator) index (0-8), -1 for not used;
        digit_positions: reference digit positions to be highlighted;
        """
        super().__init__()
        self.cell = cell
        self.host = host
        self.digit = digit
        self.box_id = box_id
        self.row_id = row_id
        self.column_id = column_id
        self.digit_positions = digit_positions
        self.priority = 2

    def __str__(self):
        return f"Digit: {self.digit}|{self.HOST_MAP[self.host]}|BOXID: {self.box_id}|ROWID: {self.row_id}|COLUMNID: {self.column_id}|POSITIONS: {self.digit_positions}"


class LastFreeHint(Hint):
    def __init__(self, host="box", cell=(0, 0), digit=0):
        """
        host: either box, row or column, the place where last remaining happens;
        index: the inderemaining cell in the host;
        TOx of the last DO: Might not have a host, need revision
        """
        super().__init__()
        self.host = host
        self.cell = cell
        self.priority = 1  # Most important hint
        self.digit = digit

    def __str__(self):
        return f"FREE|{self.HOST_MAP[self.host]}|CELL: {self.cell}"


class LastPossibleNumberHint(Hint):
    """
    Last possible number is a simple strategy that is suitable for beginners. It is based on finding the missing number. To find the missing number you should take a look at the numbers that are already exist in the 3x3 block you are interested in, and in the rows and columns connected with it. url: https://sudoku.com/sudoku-rules/last-possible-number/
    """

    def __init__(
        self,
        cell=(0, 0),
        digit=1,
        box_id=[],
        row_id=[],
        column_id=[],
        digit_positions=set(),
    ):  # Note: no host needed in this hint class
        super().__init__()
        self.cell = cell
        self.digit = digit
        self.box_id = box_id
        self.row_id = row_id
        self.column_id = column_id
        self.digit_positions = digit_positions
        self.priority = 3

    def __str__(self):
        return f"Digit: {self.digit}|BOXID: {self.box_id}|ROWID: {self.row_id}|COLUMNID: {self.column_id}|POSITIONS: {self.digit_positions}"



class NegativeHint(Hint):
    def __init__(self):
        super().__init__()
        self.negative = True
        self.priority = 0


class ObservableErrorHint(NegativeHint):
    """
    we know that the cell is wrong and also why it's wrong -- i.e. we can see a recognized digit in the same row, column or box with 100% certainty
    digit: the wrong digit that is recognized
    error_cell: the cell that is wrong
    reference_cell: the cell in the (believed to be corrected) puzzle that is the same as the wrong digit
    host: the host where the mistake is detected
    index: position of the host (0-8)
    """

    def __init__(self, digit=-1, cell=(0, 0), reference=(9, 9), host="box", index=-1):
        super().__init__()
        self.digit = digit
        self.cell = cell
        self.reference_cell = reference
        self.host = host
        self.index = index
        self.priority = -2

    def __str__(self):
        return f"CLEARLY WRONG --- Digit: {self.digit}|ERROR: {self.cell}|REFERENCE: {self.reference_cell}|{self.HOST_MAP[self.host]}|INDEX: {self.index}"


class NotObviousErrorHint(NegativeHint):
    """
    we know that the cell is wrong only because we compare it with the ground truth
    also no host at the moment, but probably hardest to deal with
    """

    def __init__(self, digit=-1, cell=(0, 0)):
        super().__init__()
        self.priority = -1
        self.digit = digit
        self.cell = cell

    def __str__(self):
        return f"WRONG --- Digit: {self.digit}|ERROR: {self.cell}"
