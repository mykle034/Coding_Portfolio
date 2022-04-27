'''
othellogame module

sets up an Othello game closely following the book's framework for games

OthelloState is a class that will handle our state representation, then we've
got stand-alone functions for player, actions, result and terminal_test

Differing from the book's framework, is that utility is *not* a stand-alone
function, as each player might have their own separate way of calculating utility


'''

print('\n Duane Myklejord, mykle034@umn.edu, 4831000 \n')

import time
import copy
import random
import numpy as np

WHITE = 1
BLACK = -1
EMPTY = 0
SIZE = 8
SKIP = "SKIP"

class RandomPlayer:
    '''Template class for an Othello Player

    An othello player *must* implement the following methods:

    get_color(self) - correctly returns the agent's color

    make_move(self, state) - given the state, returns an action that is the agent's move
    '''
    def __init__(self, mycolor):
        self.color = mycolor

    def get_color(self):
        return self.color

    #Chooses a random move from the list of legal moves
    def make_move(self, state):
        return random.choice(actions(state))


class MinimaxPlayer:
    def __init__(self, mycolor, depth):
        self.color = mycolor
        self.depth = depth

    def get_color(self):
        return self.color

    def make_move(self, state):
        display(state)
        return self.minimax_search(state)

    #Main minimax search function
    def minimax_search(self, state):
        global PLAYER

        PLAYER = player(state)

        val, move = self.max_value(state, 0)
        return move

    #Max value function for minimax
    def max_value(self, state, ply):

        if self.Is_Cutoff(state, ply):
            return self.Utility(state, PLAYER), None
        v, move = -64000, None
        for a in actions(state):
            v2, a2 = self.min_value(result(state, a), ply+1)
            if v2 > v:
                v, move = v2, a
        return v, move

    #Min Value funciton for minimax
    def min_value(self, state, ply):

        if self.Is_Cutoff(state, ply):
            return self.Utility(state, PLAYER), None
        v, move = 64000, None
        for a in actions(state):
            v2, a2 = self.max_value(result(state, a), ply+1)
            if v2 < v:
                v, move = v2, a
        return v, move

    #Utility function, higher is better
    def Utility(self, state, PLAYER):

        #Ensures a high (or low) value if a terminal state
        if self.Is_Terminal(state):
            piece_sum = sum(sum(np.absolute(state.board_array)))
            if piece_sum >= 0:
                return 64
            else:
                return -64
        else:
            return  np.sum(PLAYER.color*state.board_array[:])

    def Is_Cutoff(self, state, ply):
        '''Checks for search depth is reach, or checks for terminal state.

        '''

        #If the search depth is reached
        if ply >= self.depth:
            return True
        if self.Is_Terminal(state):
            return True
        return False

    def Is_Terminal(self, state):
        '''Simple terminal test
        '''

        # if both players have skipped
        if state.num_skips == 2:
            return True

        # if there are no empty spaces
        empty_count = 0
        for i in range(SIZE):
            for j in range(SIZE):
                if state.board_array[i][j] == EMPTY:
                    empty_count += 1
        if empty_count == 0:
            return True

        return False


#Alpha beta minimax class
class AB_MinimaxPlayer:
    def __init__(self, mycolor, depth):
        self.color = mycolor
        self.depth = depth

    def get_color(self):
        return self.color

    def make_move(self, state):
        display(state)
        return self.AB_minimax_search(state)

    # Main alpha-beta search function
    def AB_minimax_search(self, state):
        global PLAYER

        PLAYER = player(state)

        val, move = self.AB_max_value(state, 0, -64000, 64000)
        return move

    #Max value function for alpha-beta
    def AB_max_value(self, state, ply, alpha, beta):

        if self.AB_Is_Cutoff(state, ply):
            return self.AB_Utility(state, PLAYER), None
        v, move = -64000, None
        for a in actions(state):
            v2, a2 = self.AB_min_value(result(state, a), ply+1, alpha, beta)
            if v2 > v:
                v, move = v2, a
                alpha = max(alpha, v)
            if v >= beta: return v,move
        return v, move

    #Min value function for alpha-beta
    def AB_min_value(self, state, ply, alpha, beta):
        if self.AB_Is_Cutoff(state, ply):
            return self.AB_Utility(state, PLAYER), None
        v, move = 64000, None
        for a in actions(state):
            v2, a2 = self.AB_max_value(result(state, a), ply+1, alpha, beta)
            if v2 < v:
                v, move = v2, a
                beta = min(beta, v)
            if v <= alpha: return v,move
        return v, move

    #Utility function, higher is better
    def AB_Utility(self, state, PLAYER):

        #Ensures a high (or low) value if a terminal state
        if self.AB_Is_Terminal(state):
            piece_sum = sum(sum(np.absolute(state.board_array)))
            if piece_sum >= 0:
                return 64
            else:
                return -64
        else:
            return  np.sum(PLAYER.color*state.board_array[:])

    def AB_Is_Cutoff(self, state, ply):
        '''For alpha Beta -- Checks for search depth is reach, or checks for terminal state.
        '''

        #If the search depth is reached
        if ply >= self.depth:
            return True
        if self.AB_Is_Terminal(state):
            return True
        return False

    def AB_Is_Terminal(self, state):
        '''Simple terminal test
        '''

        # if both players have skipped
        if state.num_skips == 2:
            return True

        # if there are no empty spaces
        empty_count = 0
        for i in range(SIZE):
            for j in range(SIZE):
                if state.board_array[i][j] == EMPTY:
                    empty_count += 1
        if empty_count == 0:

            return True

        return False


class HumanPlayer:
    def __init__(self, mycolor):
        self.color = mycolor

    def get_color(self):
        return self.color

    def make_move(self, state):
        curr_move = None
        legals = actions(state)
        while curr_move == None:
            display(state)
            if self.color == 1:
                print("White ", end='')
            else:
                print("Black ", end='')
            print(" to play.")
            print("Legal moves are " + str(legals))
            move = input("Enter your move as a c,r pair:")
            if move == "":
                return legals[0]

            if move == SKIP and SKIP in legals:
                return move

            try:
                movetup = int(move.split(',')[0]), int(move.split(',')[1])
            except:
                movetup = None
            if movetup in legals:
                curr_move = movetup
            else:
                print("That doesn't look like a legal action to me")
        return curr_move

class OthelloState:
    '''A class to represent an othello game state'''

    def __init__(self, currentplayer, otherplayer, board_array = None, num_skips = 0):
        if board_array != None:
            self.board_array = board_array
        else:
            self.board_array = [[EMPTY] * SIZE for i in range(SIZE)]
            self.board_array[3][3] = WHITE
            self.board_array[4][4] = WHITE
            self.board_array[3][4] = BLACK
            self.board_array[4][3] = BLACK
        self.num_skips = num_skips
        self.current = currentplayer
        self.other = otherplayer


def player(state):
    return state.current

def actions(state):
    '''Return a list of possible actions given the current state
    '''
    legal_actions = []
    for i in range(SIZE):
        for j in range(SIZE):
            if result(state, (i,j)) != None:
                legal_actions.append((i,j))
    if len(legal_actions) == 0:
        legal_actions.append(SKIP)
    return legal_actions

def result(state, action):
    '''Returns the resulting state after taking the given action

    (This is the workhorse function for checking legal moves as well as making moves)

    If the given action is not legal, returns None

    '''
    # first, special case! an action of SKIP is allowed if the current agent has no legal moves
    # in this case, we just skip to the other player's turn but keep the same board
    if action == SKIP:
        newstate = OthelloState(state.other, state.current, copy.deepcopy(state.board_array), state.num_skips + 1)
        return newstate

    if state.board_array[action[0]][action[1]] != EMPTY:
        return None

    color = state.current.get_color()
    # create new state with players swapped and a copy of the current board
    newstate = OthelloState(state.other, state.current, copy.deepcopy(state.board_array))

    newstate.board_array[action[0]][action[1]] = color

    flipped = False
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    for d in directions:
        i = 1
        count = 0
        while i <= SIZE:
            x = action[0] + i * d[0]
            y = action[1] + i * d[1]
            if x < 0 or x >= SIZE or y < 0 or y >= SIZE:
                count = 0
                break
            elif newstate.board_array[x][y] == -1 * color:
                count += 1
            elif newstate.board_array[x][y] == color:
                break
            else:
                count = 0
                break
            i += 1

        if count > 0:
            flipped = True

        for i in range(count):
            x = action[0] + (i+1) * d[0]
            y = action[1] + (i+1) * d[1]
            newstate.board_array[x][y] = color

    if flipped:
        return newstate
    else:
        # if no pieces are flipped, it's not a legal move
        return None

def terminal_test(state):
    '''Simple terminal test
    '''
    # if both players have skipped
    if state.num_skips == 2:
        return True

    # if there are no empty spaces
    empty_count = 0
    for i in range(SIZE):
        for j in range(SIZE):
            if state.board_array[i][j] == EMPTY:
                empty_count += 1
    if empty_count == 0:
        return True

    return False

def display(state):
    '''Displays the current state in the terminal window
    '''
    print('  ', end='')
    for i in range(SIZE):
        print(i,end='')
    print()
    for i in range(SIZE):
        print(i, '', end='')
        for j in range(SIZE):
            if state.board_array[j][i] == WHITE:
                print('W', end='')
            elif state.board_array[j][i] == BLACK:
                print('B', end='')
            else:
                print('-', end='')
        print()

def display_final(state):
    '''Displays the score and declares a winner (or tie)
    '''
    wcount = 0
    bcount = 0
    for i in range(SIZE):
        for j in range(SIZE):
            if state.board_array[i][j] == WHITE:
                wcount += 1
            elif state.board_array[i][j] == BLACK:
                bcount += 1

    print("Black: " + str(bcount))
    print("White: " + str(wcount))
    if wcount > bcount:
        print("White wins")
        return 1
    elif wcount < bcount:
        print("Black wins")
    else:
        print("Tie")

def play_game(p1 = None, p2 = None):
    '''Plays a game with two players. By default, uses two humans
    '''
    if p1 == None:
        p1 = HumanPlayer(BLACK)
    if p2 == None:
        p2 = HumanPlayer(WHITE)


    s = OthelloState(p1, p2)
    while True:
        action = p1.make_move(s)
        if action not in actions(s):
            print("Illegal move made by Black")
            print("White wins!")
            return
        s = result(s, action)
        if terminal_test(s):
            print("Game Over")
            display(s)
            return display_final(s)
            #return
        action = p2.make_move(s)
        if action not in actions(s):
            print("Illegal move made by White")
            print("Black wins!")
            return
        s = result(s, action)
        if terminal_test(s):
            print("Game Over")
            display(s)
            return display_final(s)
            #return

def main():

    print("This program has been modified from Professor Exely's template code, with the addition of three classes:",
    "RandomPlayer, MinimaxPlayer, and AB_MinimaxPlayer.\nThe program will initially run 10 iterations of Alpha-Beta minimax (with a max search depth of 4) against a player that only does random moves.",
    "It will illustrate the move progression, and then print out the percent of games won, and the time taken.\n(This took 1 minute on my laptop).\n\nThereafter,",
    "the program will have the Alpha-Beta minimax algorithm play against a human player (using the HumanPlayer class.)")

    input("\nPress enter to start Alpha-Beta Minimax vs RandomPlayer. \n\n")


    start = time.perf_counter()
    minimax_wins = 0
    i = 0
    j = 0
    k = 0
    iterations = 10

    start = time.perf_counter()
    ab_minimax_wins = 0
    for j in range(iterations):
        if play_game(p1 = AB_MinimaxPlayer(WHITE, 4), p2 = RandomPlayer(BLACK)):
            ab_minimax_wins += 1
        print(j+1)
        print(ab_minimax_wins)
    end = time.perf_counter()
    ab_timer = end - start

    print('Aplha-Beta MiniMax beats Random ', ab_minimax_wins/(j+1)*100, 'percent of the time.')
    print('Time to run 10 cycles of Alpha-Beta Minimax (depth == 4) vs random: ', ab_timer/60, 'minutes')

    play_game(p1 = AB_MinimaxPlayer(WHITE, 4), p2 = HumanPlayer(BLACK))

if __name__ == '__main__':
    main()
