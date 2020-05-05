# Version initialize
INITIAL_MODEL_VERSION = 2
INITIAL_MEMORY_VERSION = None

# self play
EPISODES = 25
MCTS_SIMS = 50
MEMORY_SIZE = 15000
TURNS_TO_TAU0 = 20 # turn on which it starts playing deterministically
CPUCT = 1
EPSILON = 0.2
ALPHA = 0.8

# retraining
BATCH_SIZE = 256
EPOCHS = 1
REG_CONST = 0.0001
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_LOOPS = 25

HIDDEN_CNN_LAYERS = [
	{'filters': 75, 'kernel_size': (4, 4)},
	{'filters': 75, 'kernel_size': (4, 4)},
	{'filters': 75, 'kernel_size': (4, 4)},
	{'filters': 75, 'kernel_size': (4, 4)},
	{'filters': 75, 'kernel_size': (4, 4)},
	{'filters': 75, 'kernel_size': (4, 4)}
	]

# Evaluation
EVAL_EPISODES = 20
SCORING_THRESHOLD = 1.3

# Fil locations
RUN_FOLDER = './run/'
RUN_ARCHIVE_FOLDER = './old_models/'

