from game import Game, GameState


def array_to_game(x, y):
    return (x * 11) + y


def render(game):
    output = [[' ' for i in range(11)] for j in range(11)]
    for i in range(11):
        for j in range(11):
            spot = array_to_game(i, j)
            if i % 2 and not j % 2:
                # vertical
                if game.game_state.board[spot]:
                    output[i][j] = '|'
            if not i % 2 and j % 2:
                # horizontal
                if game.game_state.board[spot]:
                    output[i][j] = '-'
            if not i % 2 and not j % 2:
                # dot
                output[i][j] = 'o'
            if i % 2 and j % 2:
                # box
                if game.game_state.board[spot] == -1:
                    output[i][j] = '2'
                elif game.game_state.board[spot] == 1:
                    output[i][j] = '1'
                else:
                    output[i][j] = ' '
    for row in output:
        out = ''.join(row)
        print(out)


game = Game()

while not game.game_state.game_over:
    render(game)
    playerturn = game.game_state.player_turn
    if playerturn == 1:
        player = '1'
    else:
        player = '2'
    print('Player ' + player + ' make move')
    print('Row: ')
    row = int(input())
    print('Column: ')
    col = int(input())
    move = array_to_game(row, col)
    while move not in game.game_state.allowed_moves:
        print('Invalid move')
        print('Row: ')
        row = int(input())
        print('Column: ')
        col = int(input())
        move = array_to_game(row, col)

    state, value, game_over, _ = game.step(move)

    if game_over:
        print('Game Over')
