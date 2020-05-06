import random


def check_box(move, board):
    top = board[move - 11] != 0
    bottom = board[move + 11] != 0
    left = board[move - 1] != 0
    right = board[move + 1] != 0
    if top and bottom and left and right:
        return 1
    return 0


def check_box_made(move, board):
    if move % 11 % 2 == 0:
        # Vertical Line
        if move % 11 == 0:
            return check_box(move + 1, board)
        if move % 11 == 10:
            return check_box(move - 1, board)
        else:
            right = check_box(move + 1, board)
            left = check_box(move - 1, board)
            return left or right
    else:
        # Horizontal Line
        if move < 11:
            return check_box(move + 11, board)
        if move > 109:
            return check_box(move - 11, board)
        else:
            bottom = check_box(move + 11, board)
            top = check_box(move - 11, board)
            return bottom or top


def avoid_box(move, board):
    top = 1
    if board[move - 11] == 0:
        top = 0
    bottom = 1
    if board[move + 11] == 0:
        bottom = 0
    left = 1
    if board[move - 1] == 0:
        left = 0
    right = 1
    if board[move + 1] == 0:
        right = 0
    if (top + bottom + left + right) == 3:
        return 0
    return 1


def check_avoid_box(move, board):
    if move % 11 % 2 == 0:
        # Vertical Line
        if move % 11 == 0:
            return avoid_box(move + 1, board)
        if move % 11 == 10:
            return avoid_box(move - 1, board)
        else:
            right = avoid_box(move + 1, board)
            left = avoid_box(move - 1, board)
            return left or right
    else:
        # Horizontal Line
        if move < 11:
            return avoid_box(move + 11, board)
        if move > 109:
            return avoid_box(move - 11, board)
        else:
            bottom = avoid_box(move + 11, board)
            top = avoid_box(move - 11, board)
            return bottom or top


def make_random(state):
    return random.choice(state.allowed_moves)


def make_box(state):
    moves = state.allowed_moves
    board = state.board
    boxes = []
    for move in state.allowed_moves:
        board[move] = 1
        if check_box_made(move, board):
            boxes.append(move)
        board[move] = 0
    box = random.randint(1,1)
    if box:
        if boxes:
            return random.choice(boxes)
    return random.choice(moves)


def make_and_avoid(state):
    moves = state.allowed_moves
    board = state.board
    boxes = []
    avoid = []
    for move in state.allowed_moves:
        board[move] = 1
        if check_box_made(move, board):
            boxes.append(move)
        if check_avoid_box(move, board):
            avoid.append(move)
        board[move] = 0
    if boxes:
        return random.choice(boxes)
    if avoid:
        return random.choice(avoid)
    return random.choice(moves)
