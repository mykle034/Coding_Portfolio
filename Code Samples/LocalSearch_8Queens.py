#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 11:57:51 2021

@author: duanemyklejord
"""

print('\n Duane Myklejord, mykle034@umn.edu, 4831000 \n')

import time
import numpy as np
import pandas as pd
import random



def AttackingQueens(state):
    cost = 0

    #Going by column, from 0 to end, to find the # of attacks for the i-th queen.
    for i in range(len(state)):

        #Goes by column, from 0 to end
        for k in range(len(state)):
            if i != k and state[k] == state[i]:        #Adds cost if other queens are in the same row
                cost += 1
            elif i != k and state[k] == state[i] + (k - i):    #Adds cost if in a decresing diagonal
                cost += 1
            elif i != k and state[k] == state[i] - (k - i):    #Adds cost if in an increasing diagonal
                cost += 1

    return int(cost/2)


#Takes the current state, and returns the full board with attack values
def BoardFill(state):
    boardfill = []

    #Find the value of the moving one queen, going systematically down columns.
    #The value is saved in in a list, boardfill
    #Iterating across columns:
    for i in range(len(state)):
        boardstate = np.copy(state).tolist()

        #Iterating down rows
        for j in range(len(state)):
           boardstate[i] = j
           boardfill.append(AttackingQueens(boardstate))

           #Making the current state a high attack value to make sure it isn't chosen.
           #(And easier to see when displayed as a chess board)
           if boardstate[i] == state[i]:
               boardfill[-1] = 88

    return boardfill

#Return the indices of the target number
def FindIndices(boardfill, target):
    return[index for index, x in enumerate(boardfill) if x == target]

def FindColumn(index, size):
    return index//size

def FindRow(index, size):
    return index % size

def hillclimb_sa(state):
    start = time.perf_counter()
    plateau = 0

    #Does the algo until a solution is found (or plateau/local max).
    while AttackingQueens(state) > 0:
        next_state = np.copy(state).tolist()

        #returns the board full of hueristic (# of attacks) values
        boardfill = BoardFill(state)

        #plateau/local max check
        plateau += 1
        if plateau > 1000:
            end = time.perf_counter()
            timer = end - start
            return state, plateau, timer, 0

        #index of min hueristic value (randomly)
        next_state_index = random.choice(FindIndices(boardfill, min(boardfill)))

        #Returns the column number of the queen that's to be moved.
        column = FindColumn(next_state_index, len(state))

        #Returns the row index of where that queen will be moved.
        row = FindRow(next_state_index, len(state))
        #Moves the queen
        next_state[column] = row

        if AttackingQueens(next_state) < AttackingQueens(state):
            state = np.copy(next_state)

    end = time.perf_counter()
    timer = end - start

    return state, plateau, timer, 1


def hillclimb_fc(state):
    start = time.perf_counter()
    plateau = 0
    next_state = np.copy(state).tolist()

    #Does the algo until a solution is found (or plateau/local max).
    while AttackingQueens(state) > 0:
        next_state = np.copy(state).tolist()

        #plateau/local max cycle check
        plateau += 1
        if plateau > 1000:
           end = time.perf_counter()
           timer = end - start
           return state, plateau, timer, 0

        #Choosing a random neighbor
        while list(state) == next_state:
            column = random.randrange(0,len(state),1)
            row = random.randrange(0,len(state),1)

            #Moves the queen in column 'column' to row 'row'.
            next_state[column] = row

        #Check if the randomly chosen neighbor is better.
        #If so, this becomes the new state.
        if AttackingQueens(next_state) < AttackingQueens(state):
            state = np.copy(next_state)

    end = time.perf_counter()
    timer = end - start

    return state, plateau, timer, 1

def sim_anneal(state):
    start = time.perf_counter()
    plateau = 0
    next_state = np.copy(state).tolist()


    #Does the algo until a solution is found (or plateau/local max)
    while AttackingQueens(state) > 0:
        next_state = np.copy(state).tolist()

        #Choosing a random neighbor
        while list(state) == next_state:
            column = random.randrange(0,len(state),1)
            row = random.randrange(0,len(state),1)

            #Moves the queen in column 'column' to row 'row'.
            next_state[column] = row

        #'Temperature' schedule
        Temp = 1 / np.log(np.power(plateau+3,5))


        #Exit if the temp becomes 0
        if not Temp:
            return state, plateau, time.perf_counter()-start, 0

        #'Energy' function
        #Positive if next_state is better
        E = AttackingQueens(state) - AttackingQueens(next_state)

        #If the next state is better, it becomes the current state.
        if E > 0:
            state = np.copy(next_state)

        #If the random state is worse, it may still become the current state
        #with probability T.
        else:
            if np.random.choice([0,1], p=[1-Temp, Temp]):
                state = np.copy(next_state)

        #plateau/local max cycle check
        plateau += 1
        if plateau > 1000:
            return state, plateau, time.perf_counter()-start, 0



    end = time.perf_counter()
    timer = end - start

    return state, plateau, timer, 1


def hillclimb_sa_backwardsOKAY(state):
    start = time.perf_counter()
    plateau = 0

    while AttackingQueens(state) > 0:
        next_state = np.copy(state).tolist()

        #returns the board full of attack values
        boardfill = BoardFill(state)

        #plateau check
        plateau += 1
        if plateau > 1000:
            end = time.perf_counter()
            timer = end - start
            return state, plateau, timer, 0

        #index of min hueristic value (randomly)
        next_state_index = random.choice(FindIndices(boardfill, min(boardfill)))

        #Returns the column number of the queen that's to be moved.
        column = FindColumn(next_state_index, len(state))

        #Returns the row index of where that queen will be moved.
        row = FindRow(next_state_index, len(state))

        #Moves the queen
        next_state[column] = row
        state = np.copy(next_state)

    end = time.perf_counter()
    timer = end - start

    return state, plateau, timer, 1


def Reporting(iterations, csvs):
    start = time.perf_counter()
    success = []
    states = []
    cycles = []
    timer = []
    length = 8
    random_initial_state = [0] * length
    hc_sa = []
    hc_fc = []
    s_an = []
    hc_sa_bkok = []


    #Hill climb, steepest ascent
    for j in range(iterations):
        print('Hill climb steepest ascent iteration: ', j+1, '/', iterations)
        for i in range(length):
            random_initial_state[i] = random.randrange(0,length,1)
        hc_sa.append(hillclimb_sa(random_initial_state))
        states.append(hc_sa[-1][0])
        cycles.append(hc_sa[-1][1])
        timer.append(hc_sa[-1][2])
        success.append(hc_sa[-1][3])

    hc_sa_df = pd.DataFrame(hc_sa, columns = ['Solution or plateau state', 'Cycles', 'Run time', 'Was a valid solution returned?'])
    if csvs == 'y':
        hc_sa_df.to_csv('hc_sa.csv')

    print('Hill Climb, Steepest Ascent: ')
    print('the average number of cycles (if not plateaued): ',
          np.mean([v for i, v in enumerate(cycles) if v < 1000]))
    print('The success rate as a decimal percent: ',np.sum(success[:])/len(success))
    print('The average time to completion: ', np.mean(timer[:]))

    success = []
    states = []
    cycles = []
    timer = []

   #Hill climb, first choice
    for j in range(iterations):
        print('Hill climb first choice iteration: ', j+1, '/', iterations)
        for i in range(length):
            random_initial_state[i] = random.randrange(0,length,1)
        hc_fc.append(hillclimb_fc(random_initial_state))
        states.append(hc_fc[-1][0])
        cycles.append(hc_fc[-1][1])
        timer.append(hc_fc[-1][2])
        success.append(hc_fc[-1][3])
    hc_fc_df = pd.DataFrame(hc_fc, columns = ['Solution or plateau state', 'Cycles', 'Run time', 'Was a valid solution returned?'])
    if csvs == 'y':
            hc_fc_df.to_csv('hc_fc.csv')

    print('Hill Climb, First Chioce: ')
    print('the average number of cycles (if not plateaued): ',
          np.mean([v for i, v in enumerate(cycles) if v < 1000]))
    print('The success rate as a decimal percent: ',np.sum(success[:])/len(success))
    print('The average time to completion: ', np.mean(timer[:]))

    success = []
    states = []
    cycles = []
    timer = []

    #Simulated annealing
    for j in range(iterations):
        print('Simualted annealinng iteration: ', j+1, '/', iterations)
        for i in range(length):
            random_initial_state[i] = random.randrange(0,length,1)
        s_an.append(sim_anneal(random_initial_state))
        states.append(s_an[-1][0])
        cycles.append(s_an[-1][1])
        timer.append(s_an[-1][2])
        success.append(s_an[-1][3])
    s_an_df = pd.DataFrame(s_an, columns = ['Solution or plateau state', 'Cycles', 'Run time', 'Was a valid solution returned?'])
    if csvs == 'y':
          s_an_df.to_csv('s_an.csv')

    print('Simulated Annealing: ')
    print('the average number of cycles (if not plateaued): ',
          np.mean([v for i, v in enumerate(cycles) if v < 1000]))
    print('The success rate as a decimal percent: ',np.sum(success[:])/len(success))
    print('The average time to completion: ', np.mean(timer[:]))

    success = []
    states = []
    cycles = []
    timer = []

     #Hill climb, steepest ascent, except backwards moves are allowed
    for j in range(iterations):
        print('Hill climb steepest ascent, backwards moves allowed: ', j+1, '/', iterations)
        for i in range(length):
            random_initial_state[i] = random.randrange(0,length,1)
        hc_sa_bkok.append(hillclimb_sa_backwardsOKAY(random_initial_state))
        states.append(hc_sa_bkok[-1][0])
        cycles.append(hc_sa_bkok[-1][1])
        timer.append(hc_sa_bkok[-1][2])
        success.append(hc_sa_bkok[-1][3])
    hc_sa_bkok_df = pd.DataFrame(hc_sa_bkok, columns = ['Solution or plateau state', 'Cycles', 'Run time', 'Was a valid solution returned?'])
    if csvs == 'y':
        hc_sa_bkok_df.to_csv('hc_sa_bkok.csv')

    print('Hill Climb, Steepest Ascent, except no requirement for the next state to be better than the current: ')
    print('the average number of cycles (if not plateaued): ',
          np.mean([v for i, v in enumerate(cycles) if v < 1000]))
    print('The success rate as a decimal percent: ',np.sum(success[:])/len(success))
    print('The average time to completion: ', np.mean(timer[:]))


    end = time.perf_counter()
    timer = end - start
    print('\n Total time for ', iterations, ' cycles of each algorithm: ', timer, 'seconds')


def main():

    print('This program will run three algorithms (Hillclimb steepest ascent,',
          'hill climb firsst choice, and simulated annealing). The program can',
          'create a csv file for each algorithm, if wanted. Each row is one iteration',
          ' and includes the solution state (or plateau state),',
          'number of algorithm cycles, run time, and if a valid solution is returned (0 or 1) for that iteration',
          '\n \n Please enter the number of iterations of each algorithm you would like ran: ')


    iterations = int(input())
    csvs = input('Would you like to save the data as CSV files? Enter y for Yes: ')
    Reporting(iterations, csvs)

main()
