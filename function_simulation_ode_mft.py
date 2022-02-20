import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'

import numpy as np 
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time 
import pandas as pd
import multiprocessing as mp
from itertools import combinations
from cycler import cycler
import matplotlib as mpl
from scipy.signal import argrelextrema
import functools
import operator
from collections import Counter



def transition_rule(s1, s2):
    """states after interaction (speaker-listenser) 7 states: A, B, C, AB, AC, BC, ABC
        

    :s1: TODO
    :s2: TODO
    :returns: TODO

    """
    result = []
    if s1.islower() and s2.islower():
        sf1 = s1
        sf2 = s2
        result.append([sf1, sf2])
    elif s1.islower() and s2.isupper():
        sf1 = s1
        v = s1.upper()
        if v in s2:
            sf2 = v
        else:
            sf2 = v + s2
            sf2 = ''.join(sorted(sf2))
        result.append([sf1, sf2])
    elif s1.isupper() and s2.islower():
        sf2 = s2
        u = s2.upper()
        for v in s1:
            if v != u:
                sf1 = s1
            else:
                sf1 = v
            result.append([sf1, sf2])
    else:
        for v in s1:
            if v in s2:
                sf1 = v
                sf2 = v
            else:
                sf1 = s1
                sf2 = v + s2

            sf2 = ''.join(sorted(sf2))
            result.append([sf1, sf2])
    return result

def transition_rule_LO(s1, s2):
    """states after interaction (speaker-listenser) 7 states: A, B, C, AB, AC, BC, ABC
        

    :s1: TODO
    :s2: TODO
    :returns: TODO

    """
    result = []
    if s1.islower() and s2.islower():
        sf2 = s2
        result.append([s1, sf2])
    elif s1.islower() and s2.isupper():
        v = s1.upper()
        if v in s2:
            sf2 = v
        else:
            sf2 = v + s2
            sf2 = ''.join(sorted(sf2))
        result.append([s1, sf2])
    elif s1.isupper() and s2.islower():
        sf2 = s2
        result.append([s1, sf2])
    else:
        for v in s1:
            if v in s2:
                sf2 = v
            else:
                sf2 = v + s2
            sf2 = ''.join(sorted(sf2))
            result.append([s1, sf2])
    return result

def all_state(number_opinion):
    """TODO: Docstring for change_rule.

    :number_opinion: TODO
    :returns: TODO

    """

    single_state = [chr(state) for state in range(ord('A'), ord('A') + number_opinion)]
    possible_state = single_state.copy()
    possible_state.extend([committed.lower() for committed in single_state])
    for length in range(2, number_opinion):
        for state in combinations(single_state, length):
            possible_state.append(''.join(state))
    possible_state.append(''.join(single_state))
    return possible_state

def change_rule(number_opinion, ng_type):
    """TODO: Docstring for change_rule.

    :number_opinion: TODO
    :returns: TODO

    """
    possible_state = all_state(number_opinion)
    length = len(possible_state)
    transition_before_list = []
    transition_after_list = []
    for s1 in possible_state:
        for s2 in possible_state:
            transition_before_list.append([s1, s2])
            if ng_type == 'original':
                transition_after_list.append(transition_rule(s1, s2))
            elif ng_type == 'LO':
                transition_after_list.append(transition_rule_LO(s1, s2))
    interaction_num = len(transition_after_list)
    change_matrix = np.zeros((interaction_num, len(possible_state)))
    for i in range(interaction_num):
        transition_after = transition_after_list[i]
        transition_before = transition_before_list[i]
        len_result = len(transition_after)
        for x in transition_before:
            index = possible_state.index(x)
            change_matrix[i, index] -= 1
            
        for one_result in transition_after:
            for x in one_result:
                index = possible_state.index(x)
                change_matrix[i, index] += 1/len_result
    c_matrix = np.round(change_matrix.reshape(length, length, length).transpose(2, 0, 1) , 15)
    return c_matrix

def actual_simulation(N, interaction_number, data_point, initial_state, state_single, seed, des):
    """TODO: Docstring for actual_simulation.

    :number_opinion: TODO
    :N: TODO
    :interaction_number: TODO
    :interval: TODO
    :seed: TODO
    :pA: TODO
    :p: TODO
    :returns: TODO

    """
    interval = int(interaction_number/data_point)
    state_single_evolution = np.zeros((data_point, len(state_single)))
    state = initial_state.copy()
    random_state = np.random.RandomState(seed)
    speaker_list = random_state.choice(N, size=interaction_number, replace=True)
    #listener_list = random_state.randint(0, N-1, size=interaction_number)
    #listener_speaker_list = np.heaviside(listener_list - speaker_list, 1)
    #listener_list = np.array(listener_list + listener_speaker_list, int)
    listener_list = random_state.randint(0, N, size=interaction_number)
    listener_speaker_index = np.where((listener_list - speaker_list) == 0)[0]
    listener_speaker_list = np.random.RandomState(seed+100).randint(1, N, size = len(listener_speaker_index)) 
    listener_list[listener_speaker_index] = (listener_list[listener_speaker_index] + listener_speaker_list) % N
    select_list = random_state.random(interaction_number)  # used to select one from more than one outcomes
    for i in range(interaction_number):
        speaker = speaker_list[i]
        listener = listener_list[i]
        speaker_state = state[speaker]
        listener_state = state[listener]
        transition_state_all = transition_rule(speaker_state, listener_state)
        transition_state_length = len(transition_state_all)
        if transition_state_length == 1:
            select_state = transition_state_all[0]
        else:
            select_number = select_list[i]
            select_state = transition_state_all[int(select_number * transition_state_length)]
        state[speaker] = select_state[0]
        state[listener] = select_state[1]
        if not i % interval:
            state_counter = Counter(state)
            j = int(i // interval)
            for single, state_index in zip(state_single, range(len(state_single))):
                state_single_evolution[j, state_index] = state_counter[single]

    des_file = des + f'seed={seed}.csv'
    df_data = pd.DataFrame(np.hstack((np.arange(0, interaction_number, interval).reshape(data_point, 1), state_single_evolution)))
    df_data.to_csv(des_file, index=None, header=None)
    return None

def actual_simulation_save_all(N, interaction_number, data_point, initial_state, state_list, seed, des, ng_type):
    """TODO: Docstring for actual_simulation.

    :number_opinion: TODO
    :N: TODO
    :interaction_number: TODO
    :interval: TODO
    :seed: TODO
    :pA: TODO
    :p: TODO
    :returns: TODO

    """
    interval = int(interaction_number/data_point)
    state_all_evolution = np.zeros((data_point, len(state_list)))
    state = initial_state.copy()
    random_state = np.random.RandomState(seed)
    speaker_list = random_state.choice(N, size=interaction_number, replace=True)
    listener_list = random_state.randint(0, N-1, size=interaction_number)
    listener_speaker_list = np.heaviside(listener_list - speaker_list, 1)
    listener_list = np.array(listener_list + listener_speaker_list, int)
    select_list = random_state.random(interaction_number)  # used to select one from more than one outcomes
    for i in range(interaction_number):
        speaker = speaker_list[i]
        listener = listener_list[i]
        speaker_state = state[speaker]
        listener_state = state[listener]
        if ng_type == 'original':
            transition_state_all = transition_rule(speaker_state, listener_state)
        elif ng_type == 'LO':
            transition_state_all = transition_rule_LO(speaker_state, listener_state)

        transition_state_length = len(transition_state_all)
        if transition_state_length == 1:
            select_state = transition_state_all[0]
        else:
            select_number = select_list[i]
            select_state = transition_state_all[int(select_number * transition_state_length)]
        state[speaker] = select_state[0]
        state[listener] = select_state[1]
        if not i % interval:
            state_counter = Counter(state)
            j = int(i // interval)
            for single, state_index in zip(state_list, range(len(state_list))):
                state_all_evolution[j, state_index] = state_counter[single]

    des_file = des + f'seed={seed}_full.csv'
    df_data = pd.DataFrame(np.hstack((np.arange(0, interaction_number, interval).reshape(data_point, 1), state_all_evolution)))
    df_data.to_csv(des_file, index=None, header=None)
    return None

def mf_ode(x, t, length, c_matrix):
    """TODO: Docstring for mf_ode.

    :arg1: TODO
    :returns: TODO

    """
    x_matrix = np.dot(x.reshape(length, 1) , x.reshape(1, length))
    dxdt = np.sum(c_matrix * x_matrix, (1, 2))
    return dxdt

