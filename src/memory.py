from collections import deque
import config


class Memory:
    def __init__(self):
        self.MEMORY_SIZE = config.MEMORY_SIZE
        self.long_term_memory = deque(maxlen=config.MEMORY_SIZE)
        self.short_term_memory = deque(maxlen=config.MEMORY_SIZE)

    def add_short_mem(self, identities, state, move_values):
        for identity in identities(state, move_values):
            state = identity[0]
            MV = identity[1]
            self.short_term_memory.append({
                'board': state.board,
                'state': state,
                'id': state.id,
                'MV': MV,
                'player_turn': state.player_turn
            })

    def clear_short_mem(self):
        self.short_term_memory = deque(maxlen=config.MEMORY_SIZE)

    def add_long_mem(self):
        for i in self.short_term_memory:
            self.long_term_memory.append(i)
        self.clear_short_mem()
