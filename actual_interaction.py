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


cpu_number = 4
fontsize = 22
ticksize= 15
legendsize = 16
alpha = 0.8
lw = 3

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']) )



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
    listener_list = random_state.randint(0, N-1, size=interaction_number)
    listener_speaker_list = np.heaviside(listener_list - speaker_list, 1)
    listener_list = np.array(listener_list + listener_speaker_list, int)
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

def parallel_actual_simulation(number_opinion, N, interaction_number, data_point, seed_list, pA, p):
    """TODO: Docstring for parallel_actual_simlation.

    :arg1: TODO
    :returns: TODO

    """
    """
    nA = int(N * pA)
    nC = int(N * p)
    nB = int(N - nA - nC * (number_opinion - 2))
    state_list = all_state(number_opinion)
    state_single = state_list[0: number_opinion]
    state_committed = state_list[number_opinion: 2*number_opinion]
    state_Atilde = state_committed[2:] 
    initial_state = ['a'] * nA + ['B'] * nB  + functools.reduce(operator.iconcat, [[i] * nC for i in state_Atilde], [])
    """
    nA = int(N * pA)
    nB = int(N * p)
    xB = int(N - nA - nB)
    xA = int(N/2)
    xB = int(N/2)
    state_list = all_state(number_opinion)
    state_single = state_list[0: number_opinion]
    #initial_state = ['a'] * nA + ['b'] * nB  + ['B'] * xB 
    initial_state = ['a'] * nA + ['b'] * nB  + ['A'] * xA + ['B'] * xB 

    des = f'../data/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_p={p}/'
    if not os.path.exists(des):
        os.makedirs(des)

    p = mp.Pool(cpu_number)
    p.starmap_async(actual_simulation, [(N, interaction_number, data_point, initial_state, state_single, seed, des) for seed in seed_list]).get()
    p.close()
    p.join()
    return None

def parallel_onecommitted(number_opinion, N, interaction_number, data_point, seed_list, pA):
    """TODO: Docstring for parallel_actual_simlation.

    :arg1: TODO
    :returns: TODO

    """
    nA = int(N * pA)
    xB = int(N - nA)
    state_list = all_state(number_opinion)
    state_single = state_list[0: number_opinion]
    initial_state = ['a'] * nA + ['B'] * xB 

    des = f'../data/actual_simulation/onecommitted/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}/'
    if not os.path.exists(des):
        os.makedirs(des)

    p = mp.Pool(cpu_number)
    p.starmap_async(actual_simulation, [(N, interaction_number, data_point, initial_state, state_single, seed, des) for seed in seed_list]).get()
    p.close()
    p.join()
    return None



number_opinion = 22
N = 100000
interaction_number = 100000
N_list = np.array([1000, 5000, 10000, 50000, 100000])
N_list = np.array([1000])
interaction_number_list = N_list* 100
data_point = 1000
interval = int(interaction_number/data_point)
seed_list = np.arange(100)
pA = 0.2
p = 0.01
p_list = [0.02, 0.03, 0.04, 0.05]
p_list = [0.01, 0.02, 0.03, 0.04, 0.05]
for N, interaction_number in zip(N_list, interaction_number_list):
    for p in p_list:
        #parallel_actual_simulation(number_opinion, N, interaction_number, data_point, seed_list, pA, p)
        pass

number_opinion = 2
N = 100000
interaction_number = N * 100
data_point = 1000
pA = 0
p = 0
#parallel_actual_simulation(number_opinion, N, interaction_number, data_point, seed_list, pA, p)

number_opinion = 2
N = 500000
interaction_number = N * 100
data_point = 1000
pA = 0.3
p = 0
parallel_onecommitted(number_opinion, N, interaction_number, data_point, seed_list, pA)
