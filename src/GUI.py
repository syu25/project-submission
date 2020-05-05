import tkinter
import random
import numpy as np
from game import Game, GameState
from models import residual_CNN
from agent import Agent, Heuristic
import config
import time

prompt1 = 'Choose Players: \nNeural Net Version: 0-5\nUser:' \
          '-1\nRandom Heuristic: -2\nMakes Boxes Heuristic: ' \
          '-3\nAvoids Giving Opponent Boxes Heuristic: -4 ' \
          '\n\nPlayer 1:'
prompt2 = 'Choose Players: \nNeural Net Version: 0-5\nUser:' \
          '-1\nRandom Heuristic: -2\nMakes Boxes Heuristic: ' \
          '-3\nAvoids Giving Opponent Boxes Heuristic: -4 ' \
          '\n\nPlayer 2:'

game = Game()

player1 = None
player2 = None
row_input = -11
col_input = -11
input_number = 0



def array_to_game(x, y):
    return (x * 11) + y


def play_matches_between_networks(game, p1_version, p2_version, EPISODES, turns_to_tau0, goes_first=0):
    # player 1 agent
    if p1_version == -1:
        player1 = GuiUser('user_player1', game.state_size, game.move_size)
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
        player2 = GuiUser('user_player2', game.state_size, game.move_size)
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
            global Visual
            turn = turn + 1
            Visual.game_prompt.set(players[state.player_turn]['name'] + '\'s turn')
            Visual.game_board.set(game.render())
            Visual.window.update_idletasks()
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


def user_input(event):
    global input_number
    global Visual
    global player1
    global player2
    global row_input
    global col_input
    inp = Visual.entry.get()
    Visual.entry.delete(0, tkinter.END)
    max_version = 19
    if Visual.state == 1:
        player1 = int(inp)
        if player1 < -4 or player1 > max_version:
            Visual.game_prompt.set(prompt1 + '\nNot available Pick another Number')
            Visual.window.update_idletasks()
            return
        Visual.game_prompt.set(prompt2)
        Visual.state = 2
        Visual.window.update_idletasks()
        return
    if Visual.state == 2:
        player2 = int(inp)
        if player2 < -4 or player2 > max_version:
            Visual.game_prompt.set(prompt2 + '\nNot available Pick another Number')
            Visual.window.update_idletasks()
            return
        Visual.game_prompt.set('Starting Game')
        Visual.state = 3
        Visual.window.update_idletasks()
        play_matches_between_networks(game, player1, player2, 1, turns_to_tau0=0, goes_first=1)
        Visual.state = 1
        Visual.game_prompt.set(prompt1)
        Visual.window.update_idletasks()
        return
    if Visual.state == 3:
        input_number += 1
        Visual.input_num.set(input_number)
        row_input = int(inp)
        Visual.state = 4
        return
    if Visual.state == 4:
        input_number += 1
        Visual.input_num.set(input_number)
        col_input = int(inp)
        Visual.state = 3


class GuiUser:
    def __init__(self, name, state_size, move_size):
        self.name = name
        self.state_size = state_size
        self.move_size = move_size

    def move(self, state, tau):
        global Visual
        global row_input
        global col_input
        global input_num
        Visual.game_prompt.set(self.name + '\'s Move\nRow:')
        Visual.window.update_idletasks()
        Visual.prompt.wait_variable(Visual.input_num)
        Visual.game_prompt.set('User Move\nColumn:')
        Visual.window.update_idletasks()
        Visual.prompt.wait_variable(Visual.input_num)
        move = array_to_game(row_input, col_input)
        while move not in state.allowed_moves:
            Visual.game_prompt.set('User Move\nInvalid Move\nRow:')
            Visual.window.update_idletasks()
            Visual.window.wait_variable(Visual.input_num)
            Visual.game_prompt.set('User Move\nColumn:')
            Visual.window.update_idletasks()
            Visual.window.wait_variable(Visual.input_num)
            move = array_to_game(row_input, col_input)
        pi = np.zeros(self.move_size)
        pi[move] = 1
        value = None
        NM_value = None
        return move, pi, value, NM_value


class GUI:
    def __init__(self, game):
        self.window = tkinter.Tk()
        self.game_prompt = tkinter.StringVar()
        self.game_prompt.set(prompt1)  # game_prompt()
        self.input_num = tkinter.IntVar()
        self.input_num.set(0)
        self.game = game
        self.game_board = tkinter.StringVar()
        self.game_board.set(self.game.render())
        self.state= 1

        self.board = tkinter.Label(textvariable=self.game_board, font='Courier')

        self.prompt = tkinter.Label(textvariable=self.game_prompt, justify='left')
        self.entry = tkinter.Entry()
        self.window.bind("<Return>", user_input)
        self.board.pack()
        self.prompt.pack()
        self.entry.pack()

Visual = GUI(game)


Visual.window.mainloop()
scores, _, points, sp_scores = play_matches_between_networks(game, player1, player2, 20, turns_to_tau0=0, goes_first=0)
print('\nScores: ')
print(scores)
print('\nFirst PLAYER / Second PLAYER SCORES')
print(sp_scores)
print('Play again?')
play_again = input()

inp = ''
state = 1


iteration = 0
play_again = 'no'

while play_again != 'no':

    print('\n')
    scores, _, points, sp_scores = play_matches_between_networks(game, player1, player2, 20, turns_to_tau0=0, goes_first=0)
    print('\nScores: ')
    print(scores)
    print('\nFirst PLAYER / Second PLAYER SCORES')
    print(sp_scores)
    print('Play again?')
    play_again = input()