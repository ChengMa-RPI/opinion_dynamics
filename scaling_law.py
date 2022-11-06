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


cpu_number = 6
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

def transition_rule_modify(s1, s2):
    """states after interaction (speaker-listenser) 7 states: A, B, C, AB, AC, BC, ABC
        

    :s1: TODO
    :s2: TODO
    :returns: TODO

    """
    result = []
    if len(s1) == 1 and len(s2) == 1 and s1.upper() and s2.upper() and s2 != 'B':
        result.append([s1, s2])
    else:
        print(s1, s2)
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

def change_rule(number_opinion):
    """TODO: Docstring for change_rule.

    :number_opinion: TODO
    :returns: TODO

    """
    possible_state = all_state(number_opinion)
    transition_before_list = []
    transition_after_list = []
    for s1 in possible_state:
        for s2 in possible_state:
            transition_before_list.append([s1, s2])
            transition_after_list.append(transition_rule(s1, s2))
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
    return change_matrix

def mf_ode(x, t, length, c_matrix):
    """TODO: Docstring for mf_ode.

    :arg1: TODO
    :returns: TODO

    """
    x_matrix = np.dot(x.reshape(length, 1) , x.reshape(1, length))
    dxdt = np.sum(c_matrix * x_matrix, (1, 2))
    return dxdt

def ode_stable(number_opinion, committed_fraction, single_fraction, c_matrix):
    """TODO: Docstring for ode_stable.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    start = 0
    end = 500
    difference = 1
    single_fraction1 = single_fraction
    length = 2**number_opinion -1 + number_opinion
    mixed_fraction = np.zeros(( length-2*number_opinion))
    while (difference) > 1e-8:
        t = np.arange(start, end, 0.01)
        initial_state = np.hstack(([single_fraction1, committed_fraction, mixed_fraction]))
        result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))
        single_fraction2 = result[-1, :number_opinion]
        difference = sum(abs(result[-1, :number_opinion] - result[-1000, :number_opinion]))
        mixed_fraction = result[-1, 2*number_opinion:]
        single_fraction1 = single_fraction2
    return single_fraction2

def attractors(number_opinion, committed_fraction, single_fraction, c_matrix, des_file):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    t1 = time.time()
    attractor = ode_stable(number_opinion, committed_fraction, single_fraction, c_matrix)
    data = np.hstack((committed_fraction, single_fraction, np.round(attractor, 14)))
    df_data = pd.DataFrame(data.reshape(1, len(data)))
    df_data.to_csv(des_file, index=None, header=None, mode='a')
    t2 = time.time()
    #print(committed_fraction, t2-t1, attractor)
    return None



number_opinion = 4
committed_fraction = np.array([0.1, 0, 0.09, 0.02])
state_list = all_state(number_opinion)
single_fraction = np.array([0, 1-sum(committed_fraction), 0, 0])

length = 2**number_opinion -1 + number_opinion
coefficient = change_rule(number_opinion)
c_matrix = np.round(coefficient.reshape(length, length, length).transpose(2, 0, 1) , 15)
mixed_fraction = np.zeros(( length-2*number_opinion))
t_start = 0
t_end = 100
t = np.arange(t_start, t_end, 0.01)
initial_state = np.hstack(([single_fraction, committed_fraction, mixed_fraction]))

result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))
index = 0
upper_index  = np.triu_indices(length)
criteria = 1- 2e-1
for t in range(t_start, 10, 2):
    pos_element = []
    neg_element = []
    i = int(t/0.01)
    x = result[i]
    result_matrix = np.dot(x.reshape(length, 1) , x.reshape(1, length))
    dxdt = c_matrix * result_matrix
    dxdt_t = np.triu(dxdt[index].transpose(), k=1) + dxdt[index]
    dxdt_upper  = dxdt_t * (1 - np.tri(*dxdt_t.shape, k=-1))
    dxdt_list = np.ravel(dxdt_upper)
    #dxdt_list = dxdt_t[upper_index] 
    dxdt_length = len(dxdt_list)
    dxdt_sort = np.sort(dxdt_list)
    dxdt_sort_index = np.argsort(dxdt_list)
    dxdt_positive = dxdt_sort[dxdt_sort>0] 
    dxdt_negative = dxdt_sort[dxdt_sort<0] 
    dxdt_positive_sum = np.sum(dxdt_positive) * criteria
    dxdt_negative_sum = np.sum(dxdt_negative) * criteria
    for j in range(dxdt_length):
        if np.sum(dxdt_sort[-j:]) >= dxdt_positive_sum:
            positive_index = dxdt_sort_index[-j:]
            break
    for k in range(dxdt_length):
        if np.sum(dxdt_sort[:k]) <= dxdt_negative_sum:
            negative_index = dxdt_sort_index[:k]
            break

    if abs(dxdt_positive_sum) < abs(dxdt_negative_sum) * 0.1:
        pos_element = 'None'
    else:
        for pos in positive_index:
            element_i = pos//length
            element_j = pos%length
            pos_element.append([state_list[element_i], state_list[element_j], dxdt_list[pos]])

    if abs(dxdt_negative_sum) < abs(dxdt_positive_sum) * 0.1:
        neg_element = 'None'
    else:
        for neg in negative_index:
            element_i = neg//length
            element_j = neg%length
            neg_element.append([state_list[element_i], state_list[element_j], dxdt_list[neg]])

    print(t, pos_element, neg_element)


