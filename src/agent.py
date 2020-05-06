import numpy as np
import random
import MCTS as mc
import heuristics
import config
import time


class Heuristic:
    def __init__(self, name, state_size, move_size, heuristic):
        self.name = name
        self.state_size = state_size
        self.move_size = move_size
        self.heuristic = heuristic

    def move(self, state, tau):
        move = 0
        time.sleep(0.5)
        if self.heuristic == 'random':
            move = heuristics.make_random(state)
        if self.heuristic == 'box':
            move = heuristics.make_box(state)
        if self.heuristic == 'avoid':
            move = heuristics.make_and_avoid(state)
        pi = np.zeros(self.move_size)
        pi[move] = 1
        value = None
        NM_value = None
        return move, pi, value, NM_value


def array_to_game(x, y):
    return (x * 11) + y


class User:
    def __init__(self, name, state_size, move_size):
        self.name = name
        self.state_size = state_size
        self.move_size = move_size

    def move(self, state, tau):
        print('User Move')
        print('Row: ')
        row = int(input())
        print('Column: ')
        col = int(input())
        move = array_to_game(row, col)
        while move not in state.allowed_moves:
            print('Invalid move')
            print('Row: ')
            row = int(input())
            print('Column: ')
            col = int(input())
            move = array_to_game(row, col)
        pi = np.zeros(self.move_size)
        pi[move] = 1
        value = None
        NM_value = None
        return move, pi, value, NM_value


class Agent:
    def __init__(self, name, state_size, move_size, simulations, cpu_count, model):
        self.name = name
        self.state_size = state_size
        self.move_size = move_size
        self.cpu_count = cpu_count
        self.MCTS_simulations = simulations
        self.model = model

        self.mcts = None

        self.train_overall_loss = []
        self.train_value_loss = []
        self.train_policy_loss = []
        self.val_overall_loss = []
        self.val_value_loss = []
        self.val_policy_loss = []

    def simulate(self):
        # moves the leaf node
        leaf, value, done, backtrack = self.mcts.move_to_leaf()
        # evaluates node
        value, backtrack = self.evaluate_leaf(leaf, value, done, backtrack)
        # backfill in MCTS
        self.mcts.back_fill(leaf, value, backtrack)

    def move(self, state, tau):
        if self.mcts == None or state.id not in self.mcts.tree:
            self.build_MCTS(state)
        else:
            self.change_root(state)

        # simulate
        for simulation in range(self.MCTS_simulations):
            self.simulate()

        # Move values of sim
        pi, values = self.get_MV(1)

        # pick move
        move, value = self.choose_move(pi, values, tau)
        next_state, _, _ = state.make_move(move)

        network_value = -self.get_predictions(next_state)[0]

        return move, pi, value, network_value

    def get_predictions(self, state):
        # predict leaf
        model_input = np.array([self.model.convert_to_input(state)])
        preds = self.model.predict(model_input)
        values = preds[1]
        logits = preds[0]
        value = values[0]  # value head
        logit = logits[0]  # policy head

        # removes probabilities of moves that are not possible
        allowed = state.allowed_moves
        mask = np.ones(logit.shape, dtype=bool)
        for move in allowed:
            mask[move] = False
        for idx, m in enumerate(mask):
            if m:
                logit[idx] = -100

        odds = np.exp(logit)
        probabilities = odds / np.sum(odds)

        return (value[0], probabilities, allowed)

    def evaluate_leaf(self, leaf, value, over, backtrack):

        if over == 0:
            value, probabilities, allowed = self.get_predictions(leaf.state)

            # maybe trying to only have probs of allowed moves?
            probs = []
            for move in allowed:
                probs.append(probabilities[move])

            for idx, move in enumerate(allowed):
                new_state, _, _ = leaf.state.make_move(move)
                if new_state.id not in self.mcts.tree:
                    node = mc.Node(new_state)
                    self.mcts.add_node(node)
                else:
                    node = self.mcts.tree[new_state.id]

                new_edge = mc.Edge(leaf, node, probs[idx], move)
                leaf.edges.append((move, new_edge))

        return (value, backtrack)

    def get_MV(self, tau):
        edges = self.mcts.root.edges
        pi = np.zeros(self.move_size, dtype=np.integer)
        values = np.zeros(self.move_size, dtype=np.float32)

        for move, edge in edges:
            pi[move] = pow(edge.stats['N'], 1 / tau)
            values[move] = edge.stats['Q']

        pi = pi / (np.sum(pi) * 1.0)
        return pi, values

    def choose_move(self, pi, values, tau):
        if tau == 0:  # no longer exploring
            moves = np.argwhere(pi == max(pi))  # pick optimal moves
            move = random.choice(moves)[0]  # if multiple optimal, randomly choose
        else:  # exploring
            move_index = np.random.multinomial(1, pi)  # choose random move from exploration probs
            move = np.where(move_index == 1)[0][0]

        value = values[move]

        return move, value

    def replay(self, long_term_memory):

        for i in range(config.TRAINING_LOOPS):
            minibatch = random.sample(long_term_memory, min(config.BATCH_SIZE, len(
                long_term_memory)))  # use all samples if less than batch size

            train_states = np.array([self.model.convert_to_input(row['state']) for row in minibatch])
            train_targets = {
                'policy_head': np.array([row['MV'] for row in minibatch]),
                'value_head': np.array([row['value'] for row in minibatch])
            }

            fit = self.model.fit(train_states, train_targets, epochs=config.EPOCHS, verbose=1, validation_split=0,
                                 batch_size=32)

            self.train_overall_loss.append(round(fit.history['loss'][config.EPOCHS - 1], 4))
            self.train_policy_loss.append(round(fit.history['policy_head_loss'][config.EPOCHS - 1], 4))
            self.train_value_loss.append(round(fit.history['value_head_loss'][config.EPOCHS - 1], 4))

    def predict(self, input_to_model):
        predictions = self.model.predict(input_to_model)
        return predictions

    def build_MCTS(self, state):
        self.root = mc.Node(state)
        self.mcts = mc.MCTS(self.root, self.cpu_count)

    def change_root(self, state):
        self.mcts.root = self.mcts.tree[state.id]
