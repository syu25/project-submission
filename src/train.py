import numpy as np
import random
from importlib import reload
from game import Game
from agent import Agent
from memory import Memory
from models import residual_CNN
from functions import play_matches
import pickle
import config

np.set_printoptions(suppress=True)
game = Game()



#Load memories
if config.INITIAL_MEMORY_VERSION == None:
    memory = Memory()
else:
    print('Loading Memory' + '...')
    memory = pickle.load(open(
        config.RUN_ARCHIVE_FOLDER + "memories/memory" + str(
            config.INITIAL_MEMORY_VERSION).zfill(4) + ".p", "rb"))

#initialize models
current_NN = residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + game.grid_shape, game.move_size, config.HIDDEN_CNN_LAYERS)
best_NN = residual_CNN(config.REG_CONST, config.LEARNING_RATE, (2,) + game.grid_shape, game.move_size, config.HIDDEN_CNN_LAYERS)

#load model
if config.INITIAL_MODEL_VERSION != None:
    best_version = config.INITIAL_MODEL_VERSION
    print('Loading model ' + str(best_version) + '...')
    model_temp = best_NN.read(best_version)
    current_NN.model.set_weights(model_temp.get_weights())
    best_NN.model.set_weights(model_temp.get_weights())
else:
    best_version = 0
    best_NN.model.set_weights(current_NN.model.get_weights())  # since original is randomized weights

print('\n')
#create players
current_player = Agent('current_player', game.state_size, game.move_size, config.MCTS_SIMS, config.CPUCT, current_NN)
best_player = Agent('best_player', game.state_size, game.move_size, config.MCTS_SIMS, config.CPUCT, best_NN)
iteration = 0

while 1:
    iteration += 1
    reload(config)

    print('iteration number ' + str(iteration))

    print('Best player version ' + str(best_version))

    #self play
    print('Running self play for '+ str(config.EPISODES) + ' episodes')

    _, memory, _, _ = play_matches(best_player, best_player, config.EPISODES, turns_to_tau0=config.TURNS_TO_TAU0, memory=memory)
    print('\n')

    memory.clear_short_mem()
    if len(memory.long_term_memory) >= config.MEMORY_SIZE:
        print('Retraining...')
        current_player.replay(memory.long_term_memory)
        print('')

        if iteration % 5 == 0:
            pickle.dump(memory, open(config.RUN_FOLDER + 'memory/memory' + str(iteration).zfill(4) + '.p', 'wb'))

        # only want up to 1000 memories, chosen at random
        memory_sample = random.sample(memory.long_term_memory, min(1000, len(memory.long_term_memory)))

        #Compare best to current
        print("A challenger approaches")
        scores, _, points, sp_scores = play_matches(best_player, current_player, config.EVAL_EPISODES, turns_to_tau0=0, memory=None)
        print('\nScores: ')
        print(scores)

        print('\nFirst PLAYER / Second PLAYER SCORES')
        print(sp_scores)

        print('\n\n')
        #replace best player
        if scores['current_player'] > scores['best_player'] * config.SCORING_THRESHOLD:
            best_version += 1
            best_NN.model.set_weights(current_NN.model.get_weights())
            best_NN.write(best_version)
    else:
        print('Memory Size: ' + str(len(memory.long_term_memory)))
