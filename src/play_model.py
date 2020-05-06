import numpy as np
from game import Game, GameState
from agent import Agent, User
from memory import Memory
from models import residual_CNN
from functions import play_matches, play_matches_between_networks
import config
import pickle

game = Game()
np.set_printoptions(suppress=True)


# Load memories
if config.INITIAL_MEMORY_VERSION == None:
    memory = Memory()
else:
    print('Loading Memory' + '...')
    memory = pickle.load(open(
        config.RUN_ARCHIVE_FOLDER + "memories/memory" + str(
            config.INITIAL_MEMORY_VERSION).zfill(4) + ".p", "rb"))

# initialize model
best_NN = residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + game.grid_shape, game.move_size,
                       config.HIDDEN_CNN_LAYERS)

# load model
best_version = config.INITIAL_MODEL_VERSION
print('Loading model ' + str(best_version) + '...')
model_temp = best_NN.read(best_version)
best_NN.model.set_weights(model_temp.get_weights())


print('\n')

# create players
best_player = Agent('best_player', game.state_size, game.move_size, config.MCTS_SIMS, config.CPUCT, best_NN)
user_player = User('player1', game.state_size, game.move_size)
iteration = 0
play_again = 'yes'

while play_again != 'no':

    print('\n')
    scores, _, points, sp_scores = play_matches_between_networks(game, -1, best_version, 1, turns_to_tau0=0, goes_first=0)
    print('\nScores: ')
    print(scores)
    print('\nFirst PLAYER / Second PLAYER SCORES')
    print(sp_scores)
    print('Play again?')
    play_again = input()
