import numpy as np
from game import Game
from functions import play_matches_between_networks

game = Game()
np.set_printoptions(suppress=True)

print('Choose Players:')
print('Neural Net Version: 0-2')
print('User: -1')
print('Random Heuristic: -2')
print('Makes Boxes Heuristic: -3')
print('Avoids Giving Opponent Boxes Heuristic: -4')

max_version = 19
print('Player 1:')
player1 = int(input())
while player1 < -4 and player1 > max_version:
    print('Player 1:')
    player1 = int(input())
print('Player 2:')
player2 = int(input())
while player2 < -4 and player2 > max_version:
    print('Player 2:')
    player2 = int(input())

iteration = 0
play_again = 'yes'

while play_again != 'no':

    print('\n')
    scores, _, points, sp_scores = play_matches_between_networks(game, player1, player2, 20, turns_to_tau0=0, goes_first=0)
    print('\nScores: ')
    print(scores)
    print('\nFirst PLAYER / Second PLAYER SCORES')
    print(sp_scores)
    print('Play again?')
    play_again = input()