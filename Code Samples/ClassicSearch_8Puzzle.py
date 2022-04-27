#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 11:42:55 2021

@author: duanemyklejord
"""

print('\n Duane Myklejord, mykle034@umn.edu, 4831000 \n')

import time
import numpy as np
from queue import PriorityQueue
from itertools import count

#goal = (6,3,4,8,2,1,7,5,0)
goal = (1, 2, 3, 4, 5, 6, 7, 8, 0)

class Node:

    def __init__(self, state, parent, action, path_cost):
        self.state = state      #the state of this node
        self.parent = parent    #the node in the tree tht generated this node
        self.action = action    #The action that waas applied to the parent's state to genrate this node
        self.path_cost = path_cost    #The total cost of the path from the initial state to this node

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

'''
def DFS(init_state):

    node = Node(init_state, None, None, 0)
    frontier = []
    frontier.append(node)

    while len(frontier) != 0:
        node = frontier.pop()
        if node.state == goal: return node
        for child in get_children(node):
            if child not in frontier and child.state != node.parent:
                 frontier.append(child)




    return print('DFS search, Frontier empty')
'''

def get_children(node):
    children = []
    s = node.state
    for action in Actions(s):
        sp = Result(s, action)
        cost = node.path_cost + Path_Cost(s, action, sp)
        children.append(Node(sp, node, action, cost))

    return children

def DLS(init_state, limit):
    start = time.perf_counter()
    node = Node(init_state, None, None, 0)
    frontier = []
    frontier.append(node)
    reached = {}
    reached[init_state] = node
    action_history = []
    state_history = []


    while len(frontier) != 0:

        node = frontier.pop()


        if node.state == goal:
            action_history, state_history = History(node, init_state)
            end = time.perf_counter()
            print('The initial state is shown below. Solved in ',end - start,'[seconds] and ', len(action_history), ' moves using Depth Limited Search')
            Visualize_State(init_state)
            for i in range(len(action_history)):
                print('\n',action_history[i])
                Visualize_State(state_history[i])
            return node
        reached[node.state] = node

        for child in get_children(node):

            isNotInFrontier = child not in frontier
            isUnderLimit = child.path_cost <= limit
            isNotReached = child.state not in reached

            if isNotInFrontier and isUnderLimit and  isNotReached:
                frontier.append(child)



def IDLS(init_state):
    limit = 1
    solution = None
    while solution is None:
        solution = DLS(init_state, limit)
        limit = limit + 1

    return solution

def ASTAR(init_state, h_fnct):
    start = time.perf_counter()
    index = count(0)
    node = Node(init_state, None, None, 0)
    frontier = PriorityQueue()
    reached = {}
    frontier.put((h_fnct(node.state), next(index), node)) #(priority value, node)

    while not frontier.empty():
        node = frontier.get()[2] #[1] retrieves the node with the lowest value

        if node.state == goal:
            action_history, state_history = History(node, init_state)
            end = time.perf_counter()
            print('The initial state is shown below. Solved in ',end - start,'[seconds] and ', len(action_history), ' moves using ASTAR. and', h_fnct)
            Visualize_State(init_state)
            for i in range(len(action_history)):
                print('\n',action_history[i])
                Visualize_State(state_history[i])
            return action_history

        reached[node.state] = node

        for child in get_children(node):
            isNotReached = child.state not in reached

            if isNotReached:
                cost = h_fnct(child.state) + child.path_cost
                frontier.put((cost, next(index), child))





#Takes in the state
def num_wrong_tiles(s):
    target = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    numwrong = 0

    for i in range(len(s)):
        if s[i] != target[i]:
            numwrong += 1

    return numwrong

#Takes in state
def manhattan_distance(s):
    target = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    loc = []
    x_loc = []
    y_loc = []
    x_goal = []
    y_goal = []

    cost = []

    # Convention:
    '''
        loc
        0 1 2
        7 8 3
        6 5 4

        x_loc
        1 2 3
        1 2 3
        1 2 3

        y_loc
        3 3 3
        2 2 2
        1 1 1
    '''
    #Returns where zero is, then where one is, then where two is, and so forth.
    for i in range(len(s)):
        loc.append(s.index(i))


        #Find X location
        if loc[i] == 0 or loc[i] == 7 or loc[i] == 6:
            x_loc.append(1)
        elif loc[i] == 1 or loc[i] == 8 or loc[i] == 5:
            x_loc.append(2)
        elif loc[i] == 2 or loc[i] == 3 or loc[i] == 4:
            x_loc.append(3)

        #Find Y location
        if loc[i] == 6 or loc[i] == 5 or loc[i] == 4:
            y_loc.append(1)
        elif loc[i] == 7 or loc[i] == 8 or loc[i] == 3:
            y_loc.append(2)
        elif loc[i] == 0 or loc[i] == 1 or loc[i] == 2:
            y_loc.append(3)

        x_goal = [1, 2, 3, 3, 3, 2, 1, 1, 2]
        y_goal = [3, 3, 3, 2, 1, 1, 1, 2, 2]

    manhatten_cost = sum(cost[:])

    return manhatten_cost


#Finds possible actions based on where the 0 (blank) tile is
#With numbering starting at upper right corner going clockwise around the game board. 9 is middle
def Actions(s):
    loc = []
    actions = []
    for i in s:
        if s[i] == 0:
            loc = i;

    if loc == 0 or loc == 6 or loc == 7:
        actions.append('right')
    elif loc == 2 or loc == 3 or loc == 4:
        actions.append('left')
    else:
        actions.append('right')
        actions.append('left')
    if loc == 0 or loc == 1 or loc == 2:
        actions.append('down')
    elif loc == 6 or loc == 5 or loc == 4:
        actions.append('up')
    else:
        actions.append('up')
        actions.append('down')

    return actions

#Equals one for this homework
def Path_Cost(s, action, sp):
    return 1

def Result(s_tuple, action):
    s = np.copy(s_tuple).tolist()
    i = s.index(0)

    if action == 'right':
        if i == 0 or i == 1 or i == 7:
            s[i], s[i+1] = s[i+1], s[i]
        elif i == 6 or i == 5:
            s[i], s[i-1] = s[i-1], s[i]
        elif i == 8:
            s[i], s[3] = s[3], s[i]

    elif action == 'left':
        if i == 1 or i == 2 or i == 8:
            s[i], s[i-1] = s[i-1], s[i]
        elif i == 4 or i == 5:
            s[i], s[i+1] = s[i+1], s[i]
        elif i == 3:
            s[i], s[8] = s[8], s[i]

    elif action == 'up':
        if i == 7:
            s[i], s[0] = s[0], s[i]
        elif i == 6:
            s[i], s[i+1] = s[i+1], s[i]
        elif i == 8:
            s[i], s[1] = s[1], s[i]
        elif i == 5:
            s[i], s[8] = s[8], s[i]
        elif i == 4 or i == 3:
            s[i], s[i-1] = s[i-1], s[i]

    elif action == 'down':
        if i == 0:
            s[i], s[7] = s[7], s[i]
        elif i == 7:
            s[i], s[i-1] = s[i-1], s[i]
        elif i == 1:
            s[i], s[8] = s[8], s[i]
        elif i == 8:
            s[i], s[5] = s[5], s[i]
        elif i == 2 or i == 3:
            s[i], s[i+1] = s[i+1], s[i]

    else:
        print('Something went wrong with choosing an action in the Result function')

    return tuple(s)

def ParentFinder(node, inits):

    actions = []

    while node != inits:
        actions.append(node.action)
        node = node.parent

    return actions.reverse()
def History(node, inits):

    action_hist = []
    state_hist = []
    while node.state != inits:
        action_hist.append(node.action)
        state_hist.append(node.state)
        #Visualize(node)
        node = node.parent

    action_hist.reverse()
    state_hist.reverse()

    temp = action_hist

    #Reverses the order to be in line with the assignment parameters
    for i in range(len(action_hist)):
        if temp[i] == 'left':
            action_hist[i] = 'right'
        elif temp[i] == 'right':
            action_hist[i] = 'left'
        elif temp[i] == 'up':
            action_hist[i] = 'down'
        elif temp[i] == 'down':
            action_hist[i] = 'up'

    return action_hist, state_hist
def Visualize(s):
    if s is None: print('Nothing to Visualize')
    else:
        print('\n', s.state[0:3], '\n', [s.state[7],s.state[8],s.state[3]], '\n', s.state[-3:-6:-1])

def Visualize_State(s):
    if s is None: print('Nothing to Visualize')
    else:
        print('\n', s[0:3], '\n', [s[7],s[8],s[3]], '\n', s[-3:-6:-1])


def main():

    inits = []
    inputs = str(input('Enter Initial State (ex: 123048756):\n'))
    for i in inputs:
        inits.append(int(i))

    #Places the input into the correct order for the program
    inits[7], inits[5] = inits[5], inits[7]
    inits[4], inits[8] = inits[8], inits[4]
    inits[7], inits[3] = inits[3], inits[7]

    inits = tuple(inits)

    start = time.perf_counter()
    #ans = DFS(inits)
    ans_IDLS = IDLS(inits)
    ans_ASTAR_num_wrong_tiles = ASTAR(inits, num_wrong_tiles)
    ans_ASTAR_manhattan_distance = ASTAR(inits, manhattan_distance)
    end = time.perf_counter()

    print('To get from start to finish, do the following ', len(ans_ASTAR_num_wrong_tiles), ' moves: ')
    print(*ans_ASTAR_num_wrong_tiles, sep='\n')
    #print( '\n it took: ', end - start, '[seconds]')
    #print(start - end, ' seconds')

main()
