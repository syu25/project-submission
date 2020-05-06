import numpy as np
import config


class Node:
    def __init__(self, state):
        self.state = state
        self.player_turn = state.player_turn
        self.id = state.id
        self.edges = []

    def is_leaf(self):
        if len(self.edges):
            return False
        return True


class Edge:
    def __init__(self, parent, child, prior, move):
        self.id = parent.state.id + '|' + child.state.id
        self.parent = parent
        self.child = child
        self.player_turn = parent.state.player_turn
        self.move = move
        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': prior,
        }


# https://towardsdatascience.com/dirichlet-distribution-a82ab942a879
class MCTS:
    def __init__(self, root, cpu_count):
        self.root = root
        self.tree = {}
        self.cpu_count = cpu_count
        self.add_node(root)

    def __len__(self):
        return len(self.tree)

    def move_to_leaf(self):
        backtrack = []
        current = self.root
        done = 0
        value = 0
        while not current.is_leaf():
            maxQU = -99999
            if current == self.root:
                epsilon = config.EPSILON
                nu = np.random.dirichlet([config.ALPHA] * len(current.edges))
            else:
                epsilon = 0
                nu = [0] * len(current.edges)

            Nb = 0
            for move, edge in current.edges:
                Nb = Nb + edge.stats['N']

            for idx, (move, edge) in enumerate(current.edges):
                U = self.cpu_count * \
                    ((1 - epsilon) * edge.stats['P'] + epsilon * nu[idx]) * \
                    np.sqrt(Nb) / (1 + edge.stats['N'])

                Q = edge.stats['Q']
                if Q + U > maxQU:
                    maxQU = Q + U
                    simulation_move = move
                    simulation_edge = edge

            new_state, value, done = current.state.make_move(simulation_move)
            current = simulation_edge.child
            backtrack.append(simulation_edge)

        return current, value, done, backtrack

    def back_fill(self, leaf, value, backtrack):

        current = leaf.state.player_turn

        for edge in backtrack:
            player_turn = edge.player_turn
            if player_turn == current:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] += 1
            edge.stats['W'] += value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def add_node(self, node):
        self.tree[node.id] = node
