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


cpu_number = 4
fontsize = 22
ticksize= 15
legendsize = 16
alpha = 0.8
lw = 3

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']) )

def all_state_approximation_three(number_opinion):
    """TODO: Docstring for reduce_state.

    :number_opinion: TODO
    :returns: TODO

    """
    state = []
    for length in range(1, number_opinion+1):
        if length == 1:
            state.extend(['A', 'B', 'C', 'a', 'c'])
        elif length >1 and length <number_opinion-1:
            state.extend([i+ 'C' * (length-2) for i in ['AB', 'AC', 'BC', 'CC']])
        elif length == number_opinion-1:
            state.extend([i+ 'C' * (length-2) for i in ['AB', 'AC', 'BC']])
        elif length == number_opinion:
            state.extend(['AB' + 'C' * (length-2)])
    return state

def transition_rule_approximation_S1(s1, s2, n):
    """states after interaction (speaker-listenser) 7 states: A, B, C, AB, AC, BC, ABC
        

    :s1: TODO
    :s2: TODO
    :returns: TODO

    """
    n1 = s1.count('C')
    n2 = s2.count('C')

    result = []
    if s1.islower() and s2.islower():
        sf1 = s1
        sf2 = s2
        result.append([(sf1, sf2, 1)])
    elif s1.islower() and s2.isupper():
        sf1 = s1
        v = s1.upper()
        if v == 'A':
            if v in s2:
                sf2 = v
            else:
                sf2 = v + s2
                sf2 = ''.join(sorted(sf2))
            result.append([(sf1, sf2, 1)])
        elif v == 'C':
            if v in s2:
                sf2_1 = v
                p_1 = n2/n
                sf2 = s2 + v
                sf2_2 = ''.join(sorted(sf2))
                p_2= 1-n2/n
                result.append([(sf1, sf2_1, p_1), (sf1, sf2_2, p_2)])
            else:
                sf2 = v + s2
                sf2 = ''.join(sorted(sf2))
                result.append([(sf1, sf2, 1)])
                
    elif s1.isupper() and s2.islower():
        sf2 = s2
        u = s2.upper()
        if u == 'A':
            for v in s1:
                if v != u:
                    sf1 = s1
                else:
                    sf1 = v
                result.append([(sf1, sf2, 1)])
        elif u == 'C':
            for v in s1:
                if v != u:
                    sf1 = s1
                    result.append([(sf1, sf2, 1)])
                else:
                    sf1_1 = v
                    p_1 = 1/n
                    sf1_2 = s1
                    p_2= 1-1/n
                    result.append([(sf1_1, sf2, p_1), (sf1_2, sf2, p_2)])

    else:
        for v in s1:
            if v == 'A' or v == 'B':
                if v in s2:
                    sf1 = v
                    sf2 = v
                else:
                    sf1 = s1
                    sf2 = v + s2
                    sf2 = ''.join(sorted(sf2))
                result.append([(sf1, sf2, 1)])
            elif v == 'C':
                if v in s2:
                    sf1_1 = v
                    sf2_1 = v
                    p_1 = n2/n
                    sf1_2 = s1
                    sf2_2 = v + s2
                    sf2_2 = ''.join(sorted(sf2_2))
                    p_2 = 1-n2/n
                    result.append([(sf1_1, sf2_1, p_1), (sf1_2, sf2_2, p_2)])
                else:
                    sf1 = s1
                    sf2 = v + s2
                    sf2 = ''.join(sorted(sf2))
                    result.append([(sf1, sf2, 1)])
    return result

def change_rule_approximation_S1(number_opinion):
    """TODO: Docstring for change_rule.

    :number_opinion: TODO
    :returns: TODO

    """
    possible_state = all_state_approximation_three(number_opinion)
    length = len(possible_state)
    transition_before_list = []
    transition_after_list = []
    for s1 in possible_state:
        for s2 in possible_state:
            transition_before_list.append([s1, s2])
            transition_after_list.append(transition_rule_approximation_S1(s1, s2, number_opinion-2))
    interaction_num = len(transition_after_list)
    change_matrix = np.zeros((interaction_num, length))
    for i in range(interaction_num):
        transition_after = transition_after_list[i]
        transition_before = transition_before_list[i]
        len_result = len(transition_after)
        for x in transition_before:
            index = possible_state.index(x)
            change_matrix[i, index] -= 1
            
        for one_result in transition_after:
            for xp in one_result:
                p = xp[-1]
                if p > 0:
                    for x in xp[:2]:
                        index = possible_state.index(x)
                        change_matrix[i, index] += 1/len_result * p
    c_matrix = np.round(change_matrix.reshape(length, length, length).transpose(2, 0, 1), 16)
    return c_matrix

def mft_evolution_approximation(number_opinion, committed_fraction, single_fraction):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    length = 4 * number_opinion - 3
    c_matrix = change_rule_approximation_S1(number_opinion)
    mixed_fraction = np.zeros((length-5))
    initial_state = np.hstack(([single_fraction, committed_fraction, mixed_fraction]))
    t = np.arange(0, 1000, 0.01)
    result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))
    return result

def approximation_to_original(number_opinion, x):
    """TODO: Docstring for function.

    :number_opinion: TODO
    :x: TODO
    :returns: TODO

    """
    y = np.zeros(2 ** number_opinion - 1 + number_opinion)
    state_list = all_state(number_opinion)
    state_list_approximation = all_state_approximation_three(number_opinion)
    state_convert_index = []
    convert = []
    for i in state_list:
        new = ''
        for j in i:
            if j == 'A' or j == 'B' or j == 'a' or j == 'b':
                new += j
            elif j.islower():
                new += 'c'
            else:
                new += 'C'
        convert.append(new)
        if new == 'b':
            state_convert_index.append(-1)
        else:
            state_convert_index.append(state_list_approximation.index(new))
    for i in range(np.size(y)):
        if state_convert_index[i] == -1:
            y[i] = 0
        else:
            y[i] = x[state_convert_index[i]] / np.sum(state_convert_index[i] == np.array(state_convert_index))
    return y







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

def parallel_actual_simulation_two_opinion_switch(number_opinion, N, interaction_number, data_point, seed_list, pA, pB, switch_direction, approx_integer):
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
    if abs(xA_A_dominate - xA_B_dominate) < 1e-5:
        xA_B_dominate = 0
        xB_B_dominate = 1 - sum(committed_fraction)


    if switch_direction == 'A-B':
        switch_threshold = xB_B_dominate 
        xA_simulation = xA_A_dominate
        xB_simulation = xB_A_dominate
    elif switch_direction == 'B-A':
        switch_threshold = xA_A_dominate
        xA_simulation = xA_B_dominate
        xB_simulation = xB_B_dominate

    nA_com = round(N * pA, 10)
    nB_com = round(N * pB, 10)
    nA_uncom = round(N * xA_simulation, 10)
    nB_uncom = round(N * xB_simulation, 10)

    if approx_integer == 'round':
        nA_com = int(round(nA_com))
        nB_com = int(round(nB_com))
        nA_uncom = int(round(nA_uncom))
        nB_uncom = int(round(nB_uncom))
    elif approx_integer == 'floor':
        nA_com = int(nA_com)
        nB_com = int(nB_com)
        nA_uncom = int(nA_uncom)
        nB_uncom = int(nB_uncom) 
    elif approx_integer == 'ceil':
        nA_com = int(np.ceil(nA_com))
        nB_com = int(np.ceil(nB_com))
        nA_uncom = int(np.ceil(nA_uncom))
        nB_uncom = int(np.ceil(nB_uncom))

    nAB_uncom = N - nA_com - nB_com - nA_uncom - nB_uncom
    state_list = all_state(number_opinion)
    state_single = state_list[0: number_opinion]
    initial_state = ['a'] * nA_com + ['b'] * nB_com  + ['A'] * nA_uncom + ['B'] * nB_uncom + ['AB'] * nAB_uncom

    des = f'../data/actual_simulation/number_opinion={number_opinion}/approx_integer=' + approx_integer + f'/N={N}_pA={pA}_pB={pB}_switch_direction=' + switch_direction + '/'
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
            #print(seed, j)
            for single, state_index in zip(state_single, range(len(state_single))):
                state_single_evolution[j, state_index] = state_counter[single]
            if switch_direction == 'A-B' and state_counter['B'] >= 0.99 * switch_threshold * N and index_switch == 0:
                index_switch = j
            elif switch_direction == 'B-A' and state_counter['A'] >= 0.99 * switch_threshold * N and index_switch == 0:
                index_switch = j
            if index_switch and j -index_switch > 20: 
                index_stop = j
                break

    des_file = des + f'seed={seed}.csv'
    df_data = pd.DataFrame(np.hstack((np.arange(0, interaction_number, interval).reshape(data_point, 1), state_single_evolution))[:index_stop])
    df_data.to_csv(des_file, index=None, header=None)
    return None

def parallel_actual_simulation_multi_opinion_switch(number_opinion, N, interaction_number, data_point, seed_list, pA, p0, switch_direction, approx_integer):
    """TODO: Docstring for parallel_actual_simlation.

    :arg1: TODO
    :returns: TODO

    """
    """
    committed_fraction = np.hstack(( np.array([pA, 0]), np.ones(number_opinion - 2) * p0 ))
    xA_dominate = np.hstack(( np.array([1-sum(committed_fraction)]), np.zeros(number_opinion - 1) )) 
    xB_dominate = np.hstack(( np.array([0, 1-sum(committed_fraction)]), np.zeros(number_opinion - 2) )) 
    xC_dominate = np.hstack(( np.array([0, 0, 1-sum(committed_fraction)]), np.zeros(number_opinion - 3) )) 
    result_A_dominate = mft_evolution(number_opinion, committed_fraction, xA_dominate)[-1]
    result_B_dominate = mft_evolution(number_opinion, committed_fraction, xB_dominate)[-1]
    result_C_dominate = mft_evolution(number_opinion, committed_fraction, xC_dominate)[-1]
    """
    committed_fraction = np.array([pA, p0*(number_opinion - 2)])
    xA_dominate = np.array([1-sum(committed_fraction), 0, 0])
    xB_dominate = np.array([0, 1-sum(committed_fraction), 0])
    xC_dominate = np.array([0, 0, 1-sum(committed_fraction)])
    result_A_dominate = mft_evolution_approximation(number_opinion, committed_fraction, xA_dominate)[-1]
    result_B_dominate = mft_evolution_approximation(number_opinion, committed_fraction, xB_dominate)[-1]
    result_C_dominate = mft_evolution_approximation(number_opinion, committed_fraction, xC_dominate)[-1]
    if np.sum(np.abs(result_A_dominate - result_B_dominate)) < 1e-5:
        result_B_dominate = np.hstack((np.array([0, 1-sum(committed_fraction), 0]), committed_fraction, np.zeros(len(result_B_dominate) - 5) ))
    if switch_direction == 'A-B':
        switch_threshold = result_B_dominate[1]
        x_simulation = result_A_dominate 
    elif switch_direction == 'B-A':
        switch_threshold = result_A_dominate[0]
        x_simulation = result_B_dominate 
    elif switch_direction == 'B-C':
        switch_threshold = result_C_dominate[2]
        x_simulation = result_B_dominate 
    x_simulation = approximation_to_original(number_opinion, x_simulation)

    n_all_opinions = np.round(x_simulation * N, 10) 
    if approx_integer == 'round':
        n_all_integer = np.array( np.round(n_all_opinions), dtype=int)
    elif approx_integer == 'floor':
        n_all_integer = np.array(n_all_opinions, dtype=int)
    elif approx_integer == 'ceil':
        n_all_integer = np.array(np.ceil(n_all_opinions), dtype=int)

    if sum(n_all_integer[:-1]) <= N :
        n_all_integer[-1] = N - sum(n_all_integer[:-1])
    else:
        index = np.where(np.cumsum(n_all_integer) > N )[0]
        n_all_integer[index]  = N - sum(n_all_integer[:index-1]) 
        n_all_integer[index+1:] = 0


    state_list = all_state(number_opinion)
    initial_state = []
    for i, j in zip(state_list, n_all_integer):
            initial_state += [i] * j

    state_single = state_list[0: number_opinion]
    des = f'../data/actual_simulation/number_opinion={number_opinion}/approx_integer=' + approx_integer + f'/N={N}_pA={pA}_p0={p0}_switch_direction=' + switch_direction + '/'
    if not os.path.exists(des):
        os.makedirs(des)
    p = mp.Pool(cpu_number)
    p.starmap_async(simulation_onetime, [(N, interaction_number, data_point, initial_state, state_single, seed, des, switch_direction, switch_threshold) for seed in seed_list]).get()
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
pA = 0.3
pB = 0
#parallel_actual_simulation(number_opinion, N, interaction_number, data_point, seed_list, pA, p)

ng_type = 'original'
number_opinion = 2
switch_direction = 'B-A'
approx_integer = 'floor'

seed_list = np.arange(100) 
"""
N = 300
interaction_number = N * 100000
des = f'../data/actual_simulation/number_opinion={number_opinion}/approx_integer=' + approx_integer + f'/N={N}_pA={pA}_pB={pB}_switch_direction=' + switch_direction + '/'
seed_seen = []
for filename in os.listdir(des):
    seed_seen.append(int(filename[filename.find('=')+1:filename.find('.')]))
seed_list = np.setdiff1d(seed_list, seed_seen)
"""

N_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for N in N_list:
    interaction_number = N * 100000
    data_point = 100000
    #parallel_actual_simulation_two_opinion_switch(number_opinion, N, interaction_number, data_point, seed_list, pA, pB, switch_direction, approx_integer)
    pass


pA = 0.08
p0 = 0.01
number_opinion = 10
switch_direction = 'B-A'
approx_integer = 'floor'

N_list = [100, 200, 300]
for N in N_list:
    #parallel_actual_simulation_multi_opinion_switch(number_opinion, N, interaction_number, data_point, seed_list, pA, p0, switch_direction, approx_integer)
    pass
