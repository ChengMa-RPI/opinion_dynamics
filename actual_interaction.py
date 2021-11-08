import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'

import numpy as np 
from scipy.integrate import odeint
from scipy.optimize import fsolve, root
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
from function_simulation_ode_mft import transition_rule, all_state, change_rule, actual_simulation, mf_ode


cpu_number = 1
fontsize = 22
ticksize= 15
legendsize = 16
alpha = 0.8
lw = 3

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']) )


def mft_evolution(number_opinion, committed_fraction, single_fraction):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    length = 2**number_opinion -1 + number_opinion
    c_matrix = change_rule(number_opinion, ng_type)
    mixed_fraction = np.zeros((length-2*number_opinion))
    initial_state = np.hstack(([single_fraction, committed_fraction, mixed_fraction]))
    t = np.arange(0, 1000, 0.01)
    result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))
    return result

def mft_fixedpoint(x, number_opinion, committed_fraction, length, c_matrix):
    initial_state = np.hstack((x[:number_opinion], committed_fraction, x[number_opinion:]))
    x_matrix = np.dot(initial_state.reshape(length, 1) , initial_state.reshape(1, length))
    dxdt = np.sum(c_matrix * x_matrix, (1, 2))
    return np.hstack((dxdt[:number_opinion], sum(x) + sum(committed_fraction) - 1))

def solve_mft_fixedpoint(number_opinion, committed_fraction, xA_list):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    length = 2**number_opinion -1 + number_opinion
    c_matrix = change_rule(number_opinion, ng_type)
    fixed_point = []
    for xA in xA_list:
        xB = 1 - np.sum(committed_fraction) - xA
        mixed_fraction = 0
        x_try = np.array([xA, xB, mixed_fraction])
        solution = fsolve(mft_fixedpoint, x_try, args=(number_opinion, committed_fraction, length, c_matrix))
        if np.all(np.isclose(mft_fixedpoint(solution, number_opinion, committed_fraction, length, c_matrix), np.zeros(len(x_try) ))):
            fixed_point.append(solution)
    return fixed_point

def parallel_actual_simulation(number_opinion, N, interaction_number, data_point, seed_list, pA, p):
    """TODO: Docstring for parallel_actual_simlation.

    :arg1: TODO
    :returns: TODO

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
    """

    des = f'../data/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_p={p}/'
    if not os.path.exists(des):
        os.makedirs(des)

    p = mp.Pool(cpu_number)
    p.starmap_async(actual_simulation, [(N, interaction_number, data_point, initial_state, state_single, seed, des) for seed in seed_list]).get()
    p.close()
    p.join()
    return None

def parallel_actual_simulation_two_opinion_switch(number_opinion, N, interaction_number, data_point, seed_list, pA, pB, switch_direction):
    """TODO: Docstring for parallel_actual_simlation.

    :arg1: TODO
    :returns: TODO

    """
    committed_fraction = np.array([pA, pB])
    xA_dominate = np.array([1-sum(committed_fraction), 0])
    xB_dominate = np.array([0, 1-sum(committed_fraction)])
    result_A_dominate = mft_evolution(number_opinion, committed_fraction, xA_dominate)
    result_B_dominate = mft_evolution(number_opinion, committed_fraction, xB_dominate)
    xA_A_dominate, xB_A_dominate = result_A_dominate[-1, :2]
    xA_B_dominate, xB_B_dominate = result_B_dominate[-1, :2]
    if switch_direction == 'A-B':
        switch_threshold = xB_B_dominate 
        xA_simulation = xA_A_dominate
    elif switch_direction == 'B-A':
        switch_threshold = xA_A_dominate
        xA_simulation = xA_B_dominate
    nA_com = int(N * pA)
    nB_com = int(N * pB)
    nA_uncom = int(N * xA_simulation)
    nB_uncom = int(N - nA_com - nB_com - nA_uncom) 
    state_list = all_state(number_opinion)
    state_single = state_list[0: number_opinion]
    initial_state = ['a'] * nA_com + ['b'] * nB_com  + ['A'] * nA_uncom + ['B'] * nB_uncom

    des = f'../data/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_pB={pB}_switch_direction=' + switch_direction + '/'
    if not os.path.exists(des):
        os.makedirs(des)
    p = mp.Pool(cpu_number)
    p.starmap_async(simulation_onetime, [(N, interaction_number, data_point, initial_state, state_single, seed, des, switch_direction, switch_threshold) for seed in seed_list]).get()
    p.close()
    p.join()
    return None

def simulation_onetime(N, interaction_number, data_point, initial_state, state_single, seed, des, switch_direction, switch_threshold):
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
    index_switch = 0
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
            #print(seed, j)
            for single, state_index in zip(state_single, range(len(state_single))):
                state_single_evolution[j, state_index] = state_counter[single]
            if switch_direction == 'A-B' and state_counter['B'] > 0.95 * switch_threshold * N and index_switch == 0:
                index_switch = j
            elif switch_direction == 'B-A' and state_counter['A'] > 0.95 * switch_threshold * N and index_switch == 0:
                index_switch = j
            if index_switch and j -index_switch > 10: 
                index_stop = j
                break

    des_file = des + f'seed={seed}.csv'
    df_data = pd.DataFrame(np.hstack((np.arange(0, interaction_number, interval).reshape(data_point, 1), state_single_evolution))[:index_stop])
    df_data.to_csv(des_file, index=None, header=None)
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
pA = 0.1
p = 0.01
p_list = [0.02, 0.03, 0.04, 0.05]
p_list = [0.005]
for N, interaction_number in zip(N_list, interaction_number_list):
    for p in p_list:
        #parallel_actual_simulation(number_opinion, N, interaction_number, data_point, seed_list, pA, p)
        pass

number_opinion = 2
N = 140
interaction_number = N * 10000
data_point = 1000
pA = 0
p = 0
#parallel_actual_simulation(number_opinion, N, interaction_number, data_point, seed_list, pA, p)

ng_type = 'original'
number_opinion = 2
data_point = 10000
pA = 0.05
pB = 0
committed_fraction = np.array([pA, pB])
single_fraction = np.array([1-sum(committed_fraction), 0])
result = mft_evolution(number_opinion, committed_fraction, single_fraction)
xA = round(result[-1, 0], 3)
#parallel_actual_simulation_two_opinion(number_opinion, N, interaction_number, data_point, seed_list, pA, pB, xA)
switch_direction = 'B-A'

"""
seed_list = np.arange(100) 
N = 140
interaction_number = N * 100000
des = f'../data/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_pB={pB}_switch_direction=' + switch_direction + '/'
seed_seen = []
for filename in os.listdir(des):
    seed_seen.append(int(filename[filename.find('=')+1:filename.find('.')]))
seed_list = np.setdiff1d(seed_list, seed_seen)
"""
N_list = [60, 80, 100, 120, 140]
for N in N_list:
    interaction_number = N * 500000
    data_point = 50000
    parallel_actual_simulation_two_opinion_switch(number_opinion, N, interaction_number, data_point, seed_list, pA, pB, switch_direction)
