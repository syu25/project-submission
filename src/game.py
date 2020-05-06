import numpy as np


def array_to_game(x, y):
    return (x * 11) + y


class Game:

    def __init__(self):
        self.current_player = 1
        self.game_state = GameState(np.array([0 for n in range(121)], dtype=np.int), 1)
        self.move_space = np.array([0 for n in range(121)], dtype=np.int)
        self.grid_shape = (11, 11)
        self.input_shape = (2, 11, 11)
        self.name = 'dots-n-boxes'
        self.state_size = len(self.game_state.binary)
        self.move_size = len(self.move_space)

    def reset(self):
        self.game_state = GameState(np.array([0 for n in range(121)], dtype=np.int), 1)
        self.current_player = 1
        return self.game_state

    def step(self, move):
        moved = self.game_state.make_move(move)
        next_state = moved[0]
        value = moved[1]
        done = moved[2]
        self.game_state = next_state

        self.current_player = self.game_state.player_turn  # not correct
        info = None
        return next_state, value, done, info

    def identities(self, state, move_values):
        identities = [(state, move_values)]
        board = state.board
        MVs = move_values
        indexes = np.array([
            10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
            21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
            32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22,
            43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33,
            54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44,
            65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55,
            76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66,
            87, 86, 85, 84, 83, 82, 81, 80, 79, 78, 77,
            98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 88,
            109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99,
            120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110
        ])
        board = np.array([board[n] for n in indexes])
        MVs = np.array([MVs[n] for n in indexes])

        identities.append((GameState(board, state.player_turn), MVs))
        return identities

    def render(self):
        output = [[' ' for i in range(11)] for j in range(11)]
        for i in range(11):
            for j in range(11):
                spot = array_to_game(i, j)
                if i % 2 and not j % 2:
                    # vertical
                    if self.game_state.board[spot]:
                        output[i][j] = '|'
                if not i % 2 and j % 2:
                    # horizontal
                    if self.game_state.board[spot]:
                        output[i][j] = '-'
                if not i % 2 and not j % 2:
                    # dot
                    output[i][j] = 'o'
                if i % 2 and j % 2:
                    # box
                    if self.game_state.board[spot] == -1:
                        output[i][j] = '2'
                    elif self.game_state.board[spot] == 1:
                        output[i][j] = '1'
                    else:
                        output[i][j] = ' '
        game_rend = '    0 1 2 3 4 5 6 7 8 9 10\n\n'
        for ind,row in enumerate(output):
            if ind < 10:
                out = ' '.join(row)
                game_rend = game_rend + str(ind) + '  ' + out + '\n'
            else:
                out = ' '.join(row)
                game_rend = game_rend + str(ind) + ' ' + out + '\n'
        return game_rend


class GameState:
    def __init__(self, board, player_turn):
        self.board = board
        self.player_turn = player_turn
        self.value = self._get_value()
        self.score = self._get_score()
        self.allowed_moves = self._allowed_moves()
        self.game_over = self._check_over()
        self.id = self._state_id()
        self.binary = self._binary()

    def _allowed_moves(self):
        allowed = []
        for index in range(121):
            if index % 2:
                if self.board[index] == 0:
                    allowed.append(index)
        return allowed

    def _get_score(self):
        tmp = self.value
        return tmp[1], tmp[2]

    def _check_over(self):
        xscore = 0
        yscore = 0
        # this will be obsoleted
        for x in range(5):
            for y in range(6, 11):
                index = (y * 2) + x * 22
                if self.board[index] == self.player_turn:
                    xscore += 1
                if self.board[index] == -self.player_turn:
                    yscore += 1
        if xscore > 12:
            return 1
        if yscore > 12:
            return 1
        return 0

    def _get_value(self):
        xscore = 0
        yscore = 0
        for x in range(5):
            for y in range(6, 11):
                index = (y * 2) + x * 22
                if self.board[index] == self.player_turn:
                    xscore += 1
                if self.board[index] == -self.player_turn:
                    yscore += 1
        if yscore > 12:
            return -1, -1, 1
        if xscore > 12:
            return 1, 1, -1
        return 0, 0, 0

    def _state_id(self):
        positions1 = np.zeros(len(self.board), dtype=np.int)
        positions2 = np.zeros(len(self.board), dtype=np.int)
        for i in range(len(self.board)):
            if self.board[i] == 1:
                positions1[i] = 1
            if self.board[i] == -1:
                positions2[i] = 1
        positions = np.append(positions1, positions2)
        state_id = ''.join(map(str, positions))
        return state_id

    def _binary(self):
        positions = np.zeros(len(self.board) * 2, dtype=np.int)
        for i in range(len(self.board)):
            positions[i] = self.board[i] == 1
            positions[1 + len(self.board)] = self.board[i] == -1
        return positions

    def _check_box(self, move, board):
        top = board[move - 11] != 0
        bottom = board[move + 11] != 0
        left = board[move - 1] != 0
        right = board[move + 1] != 0
        if top and bottom and left and right:
            board[move] = self.player_turn
            return 1
        return 0

    def _check_box_made(self, move, board):
        if move % 11 % 2 == 0:
            # Vertical Line
            if move % 11 == 0:
                return self._check_box(move + 1, board)
            if move % 11 == 10:
                return self._check_box(move - 1, board)
            else:
                right = self._check_box(move + 1, board)
                left = self._check_box(move - 1, board)
                return left or right
        else:
            # Horizontal Line
            if move < 11:
                return self._check_box(move + 11, board)
            if move > 109:
                return self._check_box(move - 11, board)
            else:
                bottom = self._check_box(move + 11, board)
                top = self._check_box(move - 11, board)
                return bottom or top

    def make_move(self, move):
        board = np.array(self.board)
        board[move] = self.player_turn
        made = 0
        if self._check_box_made(move, board):
            made = 1

        state = GameState(board, -self.player_turn)
        value = 0
        over = 0

        if state.game_over:
            value = state.value[0]
            over = 1
        elif made == 1:
            state.player_turn = -state.player_turn  # double negation to keep turn

        return state, value, over
