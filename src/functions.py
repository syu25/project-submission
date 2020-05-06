import numpy as np
import random
from game import Game, GameState
from models import residual_CNN
from agent import Agent, User, Heuristic
import config


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


def play_matches_between_networks(game, p1_version, p2_version, EPISODES, turns_to_tau0, goes_first=0):
    # player 1 agent
    if p1_version == -1:
        player1 = User('user_player1', game.state_size, game.move_size)
    elif p1_version == -2:
        player1= Heuristic('random_player1', game.state_size, game.move_size, 'random')
    elif p1_version == -3:
        player1 = Heuristic('box_player1', game.state_size, game.move_size, 'box')
    elif p1_version == -4:
        player1 = Heuristic('avoid_player1', game.state_size, game.move_size, 'avoid')
    else:
        player1_NN = residual_CNN(config.REG_CONST, config.LEARNING_RATE, game.input_shape, game.move_size,
                                  config.HIDDEN_CNN_LAYERS)
        if p1_version > 0:
            p1_network = player1_NN.read(p1_version)
            player1_NN.model.set_weights(p1_network.get_weights())
        player1 = Agent('NN_player1', game.state_size, game.move_size, config.MCTS_SIMS, config.CPUCT, player1_NN)

    # player 2 agent
    if p2_version == -1:
        player2 = User('user_player2', game.state_size, game.move_size)
    elif p2_version == -2:
        player2 = Heuristic('random_player2', game.state_size, game.move_size, 'random')
    elif p2_version == -3:
        player2 = Heuristic('box_player2', game.state_size, game.move_size, 'box')
    elif p2_version == -4:
        player2 = Heuristic('avoid_player2', game.state_size, game.move_size, 'avoid')
    else:
        player2_NN = residual_CNN(config.REG_CONST, config.LEARNING_RATE, game.input_shape, game.move_size,
                                  config.HIDDEN_CNN_LAYERS)
        if p2_version > 0:
            p2_network = player2_NN.read(p2_version)
            player2_NN.model.set_weights(p2_network.get_weights())
        player2 = Agent('NN_player2', game.state_size, game.move_size, config.MCTS_SIMS, config.CPUCT, player2_NN)

    scores, memory, points, sp_scores = play_matches(player1, player2, EPISODES, turns_to_tau0, goes_first=goes_first)
    return scores, memory, points, sp_scores


def play_matches(player1, player2, EPISODES, turns_to_tau0, memory=None, goes_first=0):
    game = Game()
    scores = {player1.name: 0, "tie": 0, player2.name: 0}
    sp_scores = {'sp': 0, 'tied': 0, 'nsp': 0}
    points = {player1.name: [], player2.name: []}

    for e in range(EPISODES):
        print('Episode: ' + str(e))

        state = game.reset()

        game_over = 0
        turn = 0
        player1.mcts = None
        player2.mcts = None

        if not goes_first:
            p1_start = random.choice([-1, 1])  # either -1 or 1
        else:
            p1_start = 1
        if p1_start == 1:
            players = {
                1: {'agent': player1, 'name': player1.name},
                -1: {'agent': player2, 'name': player2.name}
            }
        else:
            players = {
                1: {'agent': player2, 'name': player2.name},
                -1: {'agent': player1, 'name': player1.name}
            }

        while not game_over:
            turn = turn + 1
            #print(game.render())

            #print('Turn: ' + str(turn))

            if turn < turns_to_tau0:
                move, pi, MCTS_val, NN_val = players[state.player_turn]['agent'].move(state, 1)
            else:
                move, pi, MCTS_val, NN_val = players[state.player_turn]['agent'].move(state, 0)

            if memory != None:
                memory.add_short_mem(game.identities, state, pi)  # game.identities is function to create the identities

            # perform move
            state, value, game_over, _ = game.step(move)

            if game_over:
                if memory != None:
                    for move in memory.short_term_memory:
                        if move['player_turn'] == state.player_turn:
                            move['value'] = value
                        else:
                            move['value'] = -value
                    memory.add_long_mem()

                if value == 1:
                    print(players[state.player_turn]['name'] + 'Won')
                    scores[players[state.player_turn]['name']] += 1
                    if state.player_turn == 1:
                        sp_scores['sp'] += 1
                    else:
                        sp_scores['nsp'] += 1

                elif value == -1:
                    print(players[-state.player_turn]['name'] + ' Won')
                    scores[players[-state.player_turn]['name']] += 1
                    if state.player_turn:
                        sp_scores['nsp'] += 1
                    else:
                        sp_scores['sp'] += 1

                else:
                    print('Tie...')
                    scores['tie'] += 1
                    sp_scores['tied'] += 1

                pts = state.score
                points[players[state.player_turn]['name']].append(pts[0])
                points[players[-state.player_turn]['name']].append(pts[0])
    return scores, memory, points, sp_scores
