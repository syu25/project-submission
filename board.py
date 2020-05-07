#Tic Tac Toe game in python

board = [' ' for x in range(82)]

def insertLetter(letter, pos):
    board[pos] = letter

def spaceIsFree(pos):
    return board[pos] == ' '

def printBoard(board):
    #Board #1,2,3
    print('|-----------|-----------|-----------|')
    print('|   |   |   |   |   |   |   |   |   |')
    print('| ' + board[1] + ' | ' + board[2] + ' | ' + board[3] + ' |' + ' ' + board[10] + ' | ' + board[11] + ' | ' + board[12] + ' |' + ' ' + board[19] + ' | ' + board[20] + ' | ' + board[21] + ' |' )
    print('|   |   |   |   |   |   |   |   |   |')
    print('|- - - - - -|- - - - - -|- - - - - -|')
    print('|   |   |   |   |   |   |   |   |   |')
    print('| ' + board[4] + ' | ' + board[5] + ' | ' + board[6] + ' |' + ' ' + board[13] + ' | ' + board[14] + ' | ' + board[15] + ' |' + ' ' + board[22] + ' | ' + board[23] + ' | ' + board[24] + ' |' )
    print('|   |   |   |   |   |   |   |   |   |')
    print('|- - - - - -|- - - - - -|- - - - - -|')
    print('|   |   |   |   |   |   |   |   |   |')
    print('| ' + board[7] + ' | ' + board[8] + ' | ' + board[9] + ' |' + ' ' + board[16] + ' | ' + board[17] + ' | ' + board[18] + ' |' + ' ' + board[25] + ' | ' + board[26] + ' | ' + board[27] + ' |' )
    print('|   |   |   |   |   |   |   |   |   |')
    #Board #4,5,6
    print('|-----------|-----------|-----------|')
    print('|   |   |   |   |   |   |   |   |   |')
    print('| ' + board[28] + ' | ' + board[29] + ' | ' + board[30] + ' |' + ' ' + board[37] + ' | ' + board[38] + ' | ' + board[39] + ' |' + ' ' + board[46] + ' | ' + board[47] + ' | ' + board[48] + ' |' )
    print('|   |   |   |   |   |   |   |   |   |')
    print('|- - - - - -|- - - - - -|- - - - - -|')
    print('|   |   |   |   |   |   |   |   |   |')
    print('| ' + board[31] + ' | ' + board[32] + ' | ' + board[33] + ' |' + ' ' + board[40] + ' | ' + board[41] + ' | ' + board[42] + ' |' + ' ' + board[49] + ' | ' + board[50] + ' | ' + board[51] + ' |' )
    print('|   |   |   |   |   |   |   |   |   |')
    print('|- - - - - -|- - - - - -|- - - - - -|')
    print('|   |   |   |   |   |   |   |   |   |')
    print('| ' + board[34] + ' | ' + board[35] + ' | ' + board[36] + ' |' + ' ' + board[43] + ' | ' + board[44] + ' | ' + board[45] + ' |' + ' ' + board[52] + ' | ' + board[53] + ' | ' + board[54] + ' |' )
    print('|   |   |   |   |   |   |   |   |   |')
    #Board #7,8,9
    print('|-----------|-----------|-----------|')
    print('|   |   |   |   |   |   |   |   |   |')
    print('| ' + board[55] + ' | ' + board[56] + ' | ' + board[57] + ' |' + ' ' + board[64] + ' | ' + board[65] + ' | ' + board[66] + ' |' + ' ' + board[73] + ' | ' + board[74] + ' | ' + board[75] + ' |' )
    print('|   |   |   |   |   |   |   |   |   |')
    print('|- - - - - -|- - - - - -|- - - - - -|')
    print('|   |   |   |   |   |   |   |   |   |')
    print('| ' + board[58] + ' | ' + board[59] + ' | ' + board[60] + ' |' + ' ' + board[67] + ' | ' + board[68] + ' | ' + board[69] + ' |' + ' ' + board[76] + ' | ' + board[77] + ' | ' + board[78] + ' |' )
    print('|   |   |   |   |   |   |   |   |   |')
    print('|- - - - - -|- - - - - -|- - - - - -|')
    print('|   |   |   |   |   |   |   |   |   |')
    print('| ' + board[61] + ' | ' + board[62] + ' | ' + board[63] + ' |' + ' ' + board[70] + ' | ' + board[71] + ' | ' + board[72] + ' |' + ' ' + board[79] + ' | ' + board[80] + ' | ' + board[81] + ' |' )
    print('|   |   |   |   |   |   |   |   |   |')
    print('|-----------|-----------|-----------|')


def isWinner(bo, le):
    eachBoard = 1
    wonBoards = []
    xWins = 0
    oWins = 0
    while(eachBoard <= 9):
        if (bo[7+(eachBoard*9-9)] == le and bo[8+(eachBoard*9-9)] == le and bo[9+(eachBoard*9-9)] == le) or (bo[4+(eachBoard*9-9)] == le and bo[5+(eachBoard*9-9)] == le and bo[6+(eachBoard*9-9)] == le) or (bo[1+(eachBoard*9-9)] == le and bo[2+(eachBoard*9-9)] == le and bo[3+(eachBoard*9-9)] == le) or (bo[1+(eachBoard*9-9)] == le and bo[4+(eachBoard*9-9)] == le and bo[7+(eachBoard*9-9)] == le) or (bo[2+(eachBoard*9-9)] == le and bo[5+(eachBoard*9-9)] == le and bo[8+(eachBoard*9-9)] == le) or (bo[3+(eachBoard*9-9)] == le and bo[6+(eachBoard*9-9)] == le and bo[9+(eachBoard*9-9)] == le) or (bo[1+(eachBoard*9-9)] == le and bo[5+(eachBoard*9-9)] == le and bo[9+(eachBoard*9-9)] == le) or (bo[3+(eachBoard*9-9)] == le and bo[5+(eachBoard*9-9)] == le and bo[7+(eachBoard*9-9)] == le):
            if le == 'O':
                oWins = oWins + 1
            else:
                xWins = xWins + 1
            wonBoards.append(eachBoard)
        eachBoard = eachBoard + 1
    possibleMoves = [x for x, letter in enumerate(board) if (letter == ' ' or letter == ' O ' or letter == ' X ') and x != 0] #Creates all moves
    for i in wonBoards:
        for all in possibleMoves:

            if(all >= (i*9-8) and (all <= i*9)):
                insertLetter('-', all)

    return ((le == 'X' and xWins == 5 ) or (le == 'O' and oWins == 5))
def playerMove():
    run = True
    while run:
        move = input('Please select a position to place an \'X\' (1-81): ')
        try:
            move = int(move)
            if move > 0 and move < 82:
                if spaceIsFree(move):
                    run = False
                    insertLetter('X', move)
                else:
                    print('Sorry, this space is occupied!')
            else:
                print('Please type a number within the range!')
        except:
            print('Please type a number!')


def compMove():

    possibleMoves = [x for x, letter in enumerate(board) if letter == ' ' and x != 0] #Creates all possible moves
    move = selectRandom(possibleMoves)

    # for let in ['O', 'X']:
    #     for i in possibleMoves:
    #         boardCopy = board[:]
    #         boardCopy[i] = let
    #         if isWinner(boardCopy, let):
    #             move = i
    #             return move
    #
    # cornersOpen = []
    # for i in possibleMoves:
    #     if i in [1,3,7,9]:
    #         cornersOpen.append(i)
    #
    # if len(cornersOpen) > 0:
    #     move = selectRandom(cornersOpen)
    #     return move
    #
    # if 5 in possibleMoves:
    #     move = 5
    #     return move
    #
    # edgesOpen = []
    # for i in possibleMoves:
    #     if i in [2,4,6,8]:
    #         edgesOpen.append(i)
    #
    # if len(edgesOpen) > 0:
    #     move = selectRandom(edgesOpen)
    #
    return move

def selectRandom(li):
    import random
    ln = len(li)
    r = random.randrange(0,ln)
    return li[r]


def isBoardFull(board):
    if board.count(' ') > 1:
        return False
    else:
        return True

def main():
    print('Welcome to Tic Tac Toe!')
    printBoard(board)

    while not(isBoardFull(board)):
        if not(isWinner(board, 'O')):
            playerMove()
            isWinner(board, 'O')

            printBoard(board)
        else:
            print('Sorry, O\'s won this time!')
            break

        if not(isWinner(board, 'X')):
            move = compMove()
            if move == 0:
                print('Tie Game!')
            else:
                insertLetter('O', move)
                print('Computer placed an \'O\' in position', move , ':')
                isWinner(board, 'O')
                printBoard(board)
        else:
            isWinner(board, 'O')
            printBoard(board)
            print('X\'s won this time! Good Job!')
            break

    if isBoardFull(board):
        isWinner(board, 'O')

        print('Tie Game!')

while True:
    answer = input('Do you want to play again? (Y/N)')
    if answer.lower() == 'y' or answer.lower == 'yes':
        board = [' ' for x in range(82)]
        print('-----------------------------------')
        main()
    else:
        break
