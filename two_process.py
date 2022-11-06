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

def transition_rule_one(s1, s2):
    """states after interaction (speaker-listenser) 7 states: A, B, C, AB, AC, BC, ABC
        

    :s1: TODO
    :s2: TODO
    :returns: TODO

    """
    result = []
    s = s1.upper() + s2.upper()
    if 'B' not in s:
        result.append([s1, s2])
    elif len(''.join(set(s)).strip('B')) == 1:
        if len(s1) == 1 and len(s2) == 1:
            result = transition_rule(s1, s2)
        elif len(s1) == 2 and len(s2) == 2:
            result = transition_rule(s1, s2)
        else:
            if len(s1) == 1 and (s1.islower() or s1 == 'B'):
                result = transition_rule(s1, s2)
            elif len(s2) == 1 and (s2.islower() or s2 == 'B'):
                result = transition_rule(s1, s2)
            else:
                result.append([s1, s2])
    else:
        result.append([s1, s2])
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

def transition_rule_approximation_three(s1, s2, n):
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

def change_rule_approximation_three(number_opinion):
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
            transition_after_list.append(transition_rule_approximation_three(s1, s2, number_opinion-2))
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

def approximation_index(number_opinion, x):
    """TODO: Docstring for approximation_index.

    :arg1: TODO
    :returns: TODO

    """
    N = number_opinion
    len_x = len(x.replace('~', ''))
    if len_x == 1:
        if x == 'A':
            index = 0
        elif x == 'B':
            index = 1
        elif x == '~A':
            index = 2
        elif x == 'a':
            index = 3
        elif x == '~a':
            index = 4
    elif len_x > 1 and len_x <N:
        if x[0] == 'A':
            if x[1] == 'B':
                index = (len_x -2) * 4 + 5 + 0
            if x[1] == '~':
                index = (len_x -2) * 4 + 5 + 1
        elif x[0] == 'B':
            index = (len_x -2) * 4 + 5 + 2
        elif x[0] == '~':
            index = (len_x -2) * 4 + 5 + 3
    elif len_x == N:
        index = (len_x - 1) * 4 
    return index

def reduce_state(number_opinion):
    """TODO: Docstring for reduce_state.

    :number_opinion: TODO
    :returns: TODO

    """
    state = []
    for length in range(1, number_opinion+1):
        if length == 1:
            state.extend(['A', 'B', '~A', 'a', '~a'])
        elif length >1 and length <number_opinion-1:
            state.extend([i+ '~A' * (length-2) for i in ['AB', 'A~A', 'B~A', '~A~A']])
        elif length == number_opinion-1:
            state.extend([i+ '~A' * (length-2) for i in ['AB', 'A~A', 'B~A']])
        elif length == number_opinion:
            state.extend(['AB' + '~A' * (length-2)])
    return state

def c_approximation(number_opinion):
    """TODO: Docstring for change_rule.

    :number_opinion: TODO
    :returns: TODO

    """
    possible_state = all_state(number_opinion)
    num_all_state = len(possible_state)
    A_tilde = possible_state[2:number_opinion]
    reduced = reduce_state(number_opinion)
    reduced_b = reduced[:4] + ['b'] + reduced[4:]
    index_list = []
    for x in possible_state:
        if x != 'b':
            for single in A_tilde:
                x = x.replace(single, '~A').replace(single.lower(), '~a')
            index_list.append(reduced.index(x))
        elif x == 'b':
            index_list.append(-1)
    num_reduce_state = len(reduced)
    transition_before_list = []
    transition_after_list = []
    for s1 in possible_state:
        for s2 in possible_state:
            transition_before_list.append([s1, s2])
            transition_after_list.append(transition_rule(s1, s2))
    interaction_num = len(transition_after_list)
    change_matrix = np.zeros((interaction_num, num_reduce_state))
    for i in range(interaction_num):
        transition_after = transition_after_list[i]
        transition_before = transition_before_list[i]
        len_result = len(transition_after)
        for x in transition_before:
            if x !='b':
                for single in A_tilde:
                    x = x.replace(single, '~A').replace(single.lower(), '~a')
                #index = approximation_index(number_opinion, x_approx)
                index = reduced.index(x)
                change_matrix[i, index] -= 1
            
        for one_result in transition_after:
            for x in one_result:
                if x !='b':
                    for single in A_tilde:
                        x = x.replace(single, '~A').replace(single.lower(), '~a')
                    #index = approximation_index(number_opinion, x_approx)
                    index = reduced.index(x)
                    change_matrix[i, index] += 1/len_result
    c = change_matrix.transpose().reshape(num_reduce_state, num_all_state, num_all_state)
    c_approx1 = np.zeros((num_reduce_state, num_reduce_state, num_all_state))
    for i in range(num_reduce_state):
        index_combine = np.where(np.array(index_list) == i)[0]
        c_approx1[:, i, :] = np.average(c[:, index_combine, :], weights=[1] * len(index_combine), axis=1)

    c_approx2 = np.zeros((num_reduce_state, num_reduce_state, num_reduce_state))
    for i in range(num_reduce_state):
        index_combine = np.where(np.array(index_list) == i)[0]
        c_approx2[:, :, i] = np.average(c_approx1[:, :, index_combine], weights=[1] * len(index_combine), axis=2)
    return c_approx2

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

def ode_stable_approximation(number_opinion, initial_state, length, c_matrix):
    """TODO: Docstring for ode_stable.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    start = 0
    end = 500
    difference = 1
    while (difference) > 1e-8:
        t = np.arange(start, end, 0.01)
        result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))
        difference = sum(abs(result[-1, :number_opinion] - result[-1000, :number_opinion]))
        initial_state = result[-1]
    return result
    
def bifurcation(number_opinion):
    """TODO: Docstring for bifurcation.

    :number_opinion: TODO
    :returns: TODO

    """
    region1 = []
    if number_opinion == 2:
        for p in np.arange(0, 0.3, 0.01):
            for q in n.arange(0, 0.3, 0.01):
                committed_fraction = np.array([p, q])
                single_fraction1 = np.array([1-p-q, 0])
                result1 = ode_stable(number_opinion, committed_fraction, single_fraction1)

                single_fraction2 = np.array([0, 1-p-q])
                result2 = ode_stable(number_opinion, committed_fraction, single_fraction2)

                if sum(abs(result2 -result1))> 1e-3:
                    region1.append([p, q])
                print(p, q, result1, result2)

    if number_opinion == 3:
        for p in np.arange(0, 0.3, 0.01):

            committed_fraction = np.array([p, p, p])
            single_fraction1 = np.array([1-3*p, 0, 0])
            result1 = ode_stable(number_opinion, committed_fraction, single_fraction1)

            single_fraction2 = np.array([0, 0, 1-3*p])
            result2 = ode_stable(number_opinion, committed_fraction, single_fraction2)

            if sum(abs(result2 -result1))> 1e-3:
                region1.append([p])
            print(p, result1, result2)


    if number_opinion > 3:
        for p in np.arange(0, 0.3, 0.01):

            committed_fraction = np.ones(number_opinion) * p
            single_fraction1 = np.hstack((1-number_opinion * p, np.zeros((number_opinion-1))))
            result1 = ode_stable(number_opinion, committed_fraction, single_fraction1)

            single_fraction2 = np.hstack((np.zeros((number_opinion-1)), 1-number_opinion * p))
            result2 = ode_stable(number_opinion, committed_fraction, single_fraction2)

            if sum(abs(result2 -result1))> 1e-3:
                region1.append([p])
            print(p, result1, result2)


                
    return region1

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

def parallel_attractors(number_opinion, committed_fraction_list, single_fraction_list, des_file):
    """TODO: Docstring for parallel_attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    length = 2**number_opinion -1 + number_opinion
    coefficient = change_rule(number_opinion)
    c_matrix = np.round(coefficient.reshape(length, length, length).transpose(2, 0, 1) , 15)
    p = mp.Pool(cpu_number)
    p.starmap_async(attractors, [(number_opinion, committed_fraction, single_fraction, c_matrix, des_file) for committed_fraction, single_fraction in zip(committed_fraction_list, single_fraction_list)]).get()
    p.close()
    p.join()
    return None

def basin_attraction(number_opinion, committed_fraction):
    """TODO: Docstring for basin_attraction.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    data = []
    uncommitted_fraction = np.round(1 - sum(committed_fraction), digit)
    length = 2**number_opinion -1 + number_opinion
    coefficient = change_rule(number_opinion)
    c_matrix = np.round(coefficient.reshape(length, length, length).transpose(2, 0, 1) , 15)
    for a in np.arange(0, uncommitted_fraction+1e-6, 0.1):
        for b in np.arange(0+1e-3, uncommitted_fraction-a+1e-6, 0.1):
            c = uncommitted_fraction-a-b
            single_fraction = np.round(np.array([a, b, c]), 3)
            attractor = ode_stable(number_opinion, committed_fraction, single_fraction, c_matrix)
            attractor = np.round(attractor, 4)
            data.append(np.hstack((single_fraction, attractor)))

    df_data = pd.DataFrame(np.vstack((data)))
    des = f'../data/num_opinion={number_opinion}/'
    #des_file = des + f'committed_fraction={committed_fraction}.csv' 
    des_file = des + f'strong_committed_fraction={committed_fraction}.csv' 
    df_data.to_csv(des_file, index=None, header=None)
    return None

def one_realization(number_opinion, committed_fraction, single_fraction):
    """TODO: Docstring for one_realization.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    start = 0
    end = 100
    dt = 0.01
    length = 2**number_opinion -1 + number_opinion
    coefficient = change_rule(number_opinion)
    c_matrix = np.round(coefficient.reshape(length, length, length).transpose(2, 0, 1) , 15)
    mixed_fraction = np.zeros(( length-2*number_opinion))
    t = np.arange(start, end, dt)
    initial_state = np.hstack(([single_fraction, committed_fraction, mixed_fraction]))
    result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))
    plt.plot(t, result[:, 0], '-', alpha=alpha, linewidth=lw, label='A', color='tab:blue')
    plt.plot(t, result[:, 1], '-.', alpha=alpha, linewidth=lw, label='B')
    """
    plt.plot(t, result[:, 2]/committed_fraction[2]**2, '--', alpha=alpha, linewidth=lw, label='C', color='tab:red')
    plt.plot(t, result[:, 3]/committed_fraction[3]**2, '--', alpha=alpha, linewidth=lw, label='D')
    plt.plot(t, result[:, 4]/committed_fraction[4]**2, '--', alpha=alpha, linewidth=lw, label='E')
    plt.plot(t, result[:, 5]/committed_fraction[5]**2, '--', alpha=alpha, linewidth=lw, label='F')
    #plt.plot(t, result[:, 3], '--', alpha=alpha, linewidth=lw, label='D')
    #plt.plot(t, np.sum(result[:, number_opinion*2:], 1), '--', alpha=alpha, linewidth=lw, label='mix')
    plt.ylabel('$x$', fontsize=fontsize)
    """

    #plt.plot(t[1:], np.diff(result[:, 0])/committed_fraction[0]**2/dt, '-', alpha=alpha, linewidth=lw, label='A', color='tab:blue')
    #plt.plot(t[1:], np.diff(result[:, 1])/dt, '-.', alpha=alpha, linewidth=lw, label='B')
    #plt.plot(t[1:], np.diff(result[:, 2])/committed_fraction[2]**2/dt, '--', alpha=alpha, linewidth=lw, label='C', color='tab:red')
    #plt.plot(t[1:], np.diff(result[:, 3])/committed_fraction[3]**2/dt, '--', alpha=alpha, linewidth=lw, label='D')
    #plt.plot(t[1:], np.diff(result[:, 4])/committed_fraction[4]**2/dt, '--', alpha=alpha, linewidth=lw, label='E')
    #plt.plot(t[1:], np.diff(result[:, 5])/committed_fraction[5]**2/dt, '--', alpha=alpha, linewidth=lw, label='F')
    #plt.plot(t[1:], np.diff(np.sum(result[:, number_opinion*2:], 1))/dt, '--', alpha=alpha, linewidth=lw, label='mix')
    plt.ylabel('$x$', fontsize=fontsize)

    #plt.plot(t[1:], np.diff(result[:, 6]), '--', alpha=alpha, linewidth=lw, label='AB')
    #plt.plot(t[1:], np.diff(result[:, 7]), '--', alpha=alpha, linewidth=lw, label='AC')
    #plt.plot(t[1:], np.diff(result[:, 8]), '--', alpha=alpha, linewidth=lw, label='BC')
    #plt.plot(t[1:], np.diff(result[:, 9]), '--', alpha=alpha, linewidth=lw, label='ABC')
    #plt.plot(t, result[:, 3], '--', alpha=alpha, linewidth=lw, label='D')
    #plt.plot(t, result[:, 4], '--', alpha=alpha, linewidth=lw, label='E')
    plt.xlabel('$t$', fontsize=fontsize)
    plt.subplots_adjust(left=0.20, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.show()

    return result[:, :3]

def pA_BC(number_opinion):
    """TODO: only one committed nodes p_A, x_B and x_C are set the same initially.
    :returns: TODO

    """
    pA_list = np.arange(0, 0.2, 0.002)
    attractors = np.zeros((len(pA_list), 3))
    for pA, i in zip(pA_list, range(len(pA_list))):
        committed_fraction = [pA, 0, 0]
        uncommitted_fraction = 1 - pA
        length = 2**number_opinion -1 + number_opinion
        coefficient = change_rule(number_opinion)
        c_matrix = np.round(coefficient.reshape(length, length, length).transpose(2, 0, 1) , 15)
        a = 0
        b = uncommitted_fraction
        c = 0 
        b = uncommitted_fraction/2 + 0
        c = uncommitted_fraction/2 - 0
        bc_diff = round(b-c, 2)
        single_fraction = np.array([a, b, c])
        attractor = ode_stable(number_opinion, committed_fraction, single_fraction, c_matrix)
        attractors[i] = np.round(attractor, 4)
    plt.plot(pA_list, attractors[:, 0]+pA_list, '-.', label=f'$x_B=x_C$', alpha = alpha, linewidth=lw)
    plt.xlabel('$p_A$', fontsize=fontsize)
    plt.ylabel('$x_A^s$', fontsize=fontsize)
    plt.subplots_adjust(left=0.17, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(nbins=5)
    plt.legend(frameon=False, fontsize = legendsize, loc='lower right')
    #plt.show()
    return attractors

def attractors_approximation(number_opinion, committed_fraction, length, coefficient, des_file):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    t1 = time.time()
    initial_state = np.zeros((length))
    initial_state[3:5] = committed_fraction  
    initial_state[1] = 1- sum(committed_fraction)
    t = np.arange(0, 500, 0.01)
    #result = odeint(mf_ode, initial_state, t, args=(length, coefficient))[-1, :3]
    result = ode_stable_approximation(number_opinion, initial_state, length, coefficient)[-1, :3]
    data = np.hstack((committed_fraction, result))
    df_data = pd.DataFrame(data.reshape(1, len(data)))
    df_data.to_csv(des_file, index=None, header=None, mode='a')
    t2 = time.time()
    #print(committed_fraction, t2-t1)
    return None

def parallel_attractors_approximation(number_opinion, committed_fraction_list, des_file):
    """TODO: Docstring for parallel_attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    #coefficient = c_approximation(number_opinion)
    coefficient = change_rule_approximation_three(number_opinion)
    length = np.size(coefficient, 0)
    p = mp.Pool(cpu_number)
    p.starmap_async(attractors_approximation, [(number_opinion, committed_fraction, length, coefficient, des_file) for committed_fraction in committed_fraction_list]).get()
    p.close()
    p.join()
    return None

def approximation_oneuncommitted(number_opinion, pA_list, p_not_A_list):
    """reduce some variables, for three-opinion variant, there are 6 variables: A, A', pA, AA', A'A', AA'A' 

    :number_opinion: TODO
    :: TODO
    :returns: TODO

    """
    des = '../data/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'num_opinion={number_opinion}_oneuncommitted_approximation.csv'
    committed_fraction_list = []
    for pA, i in zip(pA_list, range(len(pA_list))):
        for p_not_A, j in zip(p_not_A_list, range(len(p_not_A_list))):
            committed_fraction_list.append([pA, p_not_A])
    parallel_attractors_approximation(number_opinion, committed_fraction_list, des_file)
    return None

def fluctuate_oneuncommitted(number_opinion, cA_list, p, sigma, seed_list, normalization):
    """TODO: introduce fluctuations to the initial committed fractions p.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    des = f'../data/num_opinion={number_opinion}_absolute/'
    if not os.path.exists(des):
        os.makedirs(des)

    for seed in seed_list:
        print(seed)
        committed_fraction_list = []
        single_fraction_list = []
        p_fluctuate = np.random.RandomState(seed).normal(0, sigma, number_opinion-2)
        cA_tilde = p + p_fluctuate 
        if normalization: 
            cA_tilde = cA_tilde/ sum(cA_tilde) * p * (number_opinion-2)
            des_file = des  + f'oneuncommitted_p={p}_sigma={sigma}_seed={seed}_normalization.csv'
        else:
            des_file = des + f'oneuncommitted_p={p}_sigma={sigma}_seed={seed}.csv'
        for cA in cA_list:
            committed_fraction = np.round(np.hstack((cA, cA_tilde, 0)), 14)
            uC = 1 - sum(committed_fraction)
            single_fraction = np.round(np.hstack((0 * np.ones(number_opinion - 1), uC )), 14)
            committed_fraction_list.append(committed_fraction)
            single_fraction_list.append(single_fraction)
        parallel_attractors(number_opinion, committed_fraction_list, single_fraction_list, des_file)
    return None

def reduce_state_two(number_opinion):
    """TODO: Docstring for reduce_state.

    :number_opinion: TODO
    :returns: TODO

    """
    state = []
    for length in range(1, number_opinion+1):
        if length == 1:
            state.extend(['A', '~A', '~a'])
        elif length >1 and length <= number_opinion-1:
            state.extend([i+ '~A' * (length-1) for i in ['A', '~A']])
        elif length == number_opinion:
            state.extend(['A' + '~A' * (length-1)])
    return state

def c_approximation_two(number_opinion):
    """TODO: Docstring for change_rule.

    :number_opinion: TODO
    :returns: TODO

    """
    possible_state = all_state(number_opinion)
    num_all_state = len(possible_state)
    A_tilde = possible_state[1:number_opinion]
    reduced = reduce_state_two(number_opinion)
    index_list = []
    for x in possible_state:
        if x != 'a':
            for single in A_tilde:
                x = x.replace(single, '~A').replace(single.lower(), '~a')
            index_list.append(reduced.index(x))
        elif x == 'a':
            index_list.append(-1)
    num_reduce_state = len(reduced)
    transition_before_list = []
    transition_after_list = []
    for s1 in possible_state:
        for s2 in possible_state:
            transition_before_list.append([s1, s2])
            transition_after_list.append(transition_rule(s1, s2))
    interaction_num = len(transition_after_list)
    change_matrix = np.zeros((interaction_num, num_reduce_state))
    for i in range(interaction_num):
        transition_after = transition_after_list[i]
        transition_before = transition_before_list[i]
        len_result = len(transition_after)
        for x in transition_before:
            if x !='a':
                for single in A_tilde:
                    x = x.replace(single, '~A').replace(single.lower(), '~a')
                #index = approximation_index(number_opinion, x_approx)
                index = reduced.index(x)
                change_matrix[i, index] -= 1
            
        for one_result in transition_after:
            for x in one_result:
                if x !='a':
                    for single in A_tilde:
                        x = x.replace(single, '~A').replace(single.lower(), '~a')
                    #index = approximation_index(number_opinion, x_approx)
                    index = reduced.index(x)
                    change_matrix[i, index] += 1/len_result
    c = change_matrix.transpose().reshape(num_reduce_state, num_all_state, num_all_state)
    c_approx1 = np.zeros((num_reduce_state, num_reduce_state, num_all_state))
    for i in range(num_reduce_state):
        index_combine = np.where(np.array(index_list) == i)[0]
        c_approx1[:, i, :] = np.average(c[:, index_combine, :], weights=[1] * len(index_combine), axis=1)

    c_approx2 = np.zeros((num_reduce_state, num_reduce_state, num_reduce_state))
    for i in range(num_reduce_state):
        index_combine = np.where(np.array(index_list) == i)[0]
        c_approx2[:, :, i] = np.average(c_approx1[:, :, index_combine], weights=[1] * len(index_combine), axis=2)
    return c_approx2

def attractors_approximation_two(number_opinion, committed_fraction, length, coefficient, des_file):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    t1 = time.time()
    initial_state = np.zeros((length))
    initial_state[2] = committed_fraction  
    initial_state[0] = 1- committed_fraction
    t = np.arange(0, 500, 0.01)
    result = odeint(mf_ode, initial_state, t, args=(length, coefficient))[-1]
    data = np.hstack((committed_fraction, result))
    df_data = pd.DataFrame(data.reshape(1, len(data)))
    df_data.to_csv(des_file, index=None, header=None, mode='a')
    t2 = time.time()
    #print(committed_fraction, t2-t1)
    return None

def parallel_attractors_approximation_two(number_opinion, committed_fraction_list, des_file):
    """TODO: Docstring for parallel_attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    coefficient = c_approximation_two(number_opinion)
    length = np.size(coefficient, 0)
    p = mp.Pool(cpu_number)
    p.starmap_async(attractors_approximation_two, [(number_opinion, committed_fraction, length, coefficient, des_file) for committed_fraction in committed_fraction_list]).get()
    p.close()
    p.join()
    return None

def approximation_oneuncommitted_two(number_opinion, p_not_A_list):
    """reduce some variables, for three-opinion variant, there are 6 variables: A, A', pA, AA', A'A', AA'A' 

    :number_opinion: TODO
    :: TODO
    :returns: TODO

    """
    des = '../data/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'num_opinion={number_opinion}_oneuncommitted_approximation_two.csv'
    committed_fraction_list = p_not_A_list
    parallel_attractors_approximation_two(number_opinion, committed_fraction_list, des_file)
    return None

def parallel_attractors_lowerbound(number_opinion_list, committed_fraction_list, des_file):
    """TODO: Docstring for parallel_attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    number_opinion_unique = np.unique(number_opinion_list)
    coefficient_list = []
    length_list = []
    for number_opinion in number_opinion_unique:
        coefficient = c_approximation(number_opinion)
        length = np.size(coefficient, 0)
        coefficient_list.append(coefficient)
        length_list.append(length)
    p = mp.Pool(cpu_number)
    p.starmap_async(attractors_approximation, [(number_opinion, committed_fraction, length_list[np.where(number_opinion == number_opinion_unique)[0][0]], coefficient_list[np.where(number_opinion == number_opinion_unique)[0][0]], des_file) for number_opinion, committed_fraction in zip(number_opinion_list, committed_fraction_list)]).get()
    p.close()
    p.join()
    return None

def fluctuate_lowerbound(number_opinion, p):
    """TODO: introduce fluctuations to the initial committed fractions p.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    des = f'../data/num_opinion={number_opinion}_lowerbound/'
    des_file = des  + f'oneuncommitted_p={p}.csv'
    if not os.path.exists(des):
        os.makedirs(des)
    committed_fraction_list = []
    single_fraction_list = []
    number_opinion_list = []
    p_cAtilde = p * (number_opinion-2)
    pmax_list = np.arange(p, p_cAtilde, 0.0001)
    for pmax in pmax_list:
        n_max = int(np.floor(p_cAtilde/pmax))
        p_cCtilde = pmax + pmax * n_max
        p_cC = p_cAtilde - pmax * n_max

        committed_fraction = np.round(np.hstack((p_cC, p_cCtilde)), 14)
        committed_fraction_list.append(committed_fraction)
        number_opinion_list.append(n_max + 3)

    parallel_attractors_lowerbound(number_opinion_list, committed_fraction_list, des_file)
    return None

def fluctuate_lowerbound2(number_opinion, p):
    """TODO: Docstring for fluctuate_lowerbound2.

    :number_opinion: TODO
    :p: TODO
    :returns: TODO

    """
    des = f'../data/lowerbound2/'
    des_file = des  + f'oneuncommitted_p={p}.csv'
    if not os.path.exists(des):
        os.makedirs(des)
    committed_fraction_list = []
    single_fraction_list = []
    p_cAtilde = p * (number_opinion-2)
    p_cA_list = np.arange(0, 0.1, 0.0001)
    for p_cA in p_cA_list:

        committed_fraction = np.round(np.hstack((p_cA, 0, p_cAtilde)), 14)
        single_fraction = np.hstack((0, 1-sum(committed_fraction), 0))
        committed_fraction_list.append(committed_fraction)
        single_fraction_list.append(single_fraction)

    parallel_attractors(number_opinion, committed_fraction_list, single_fraction_list, des_file)
    return None

def five_lowerbound(number_opinion, p):
    """TODO: Docstring for original_lowerbound.

    :number_opinion: TODO
    :returns: TODO

    """
    number_opinion = 5
    des = f'../data/num_opinion={number_opinion}_original_lowerbound/'
    if not os.path.exists(des):
        os.makedirs(des)
    #des_file = des + f'p={p}.csv'
    des_file = des + f'p={p}_zoomin2.csv'
    committed_fraction_list = []
    single_fraction_list = []
    p_cAtilde = (number_opinion - 2) * p
    p_cA_list = np.arange(0.0757+ 1e-7, 0.0758, 0.00001)
    p_cC_list = np.arange(0.0757, 0.0758, 0.00001)
    p_cD_list = np.arange(0, 0.1, 0.00001)
    for p_cC in p_cC_list:
        print(p_cC)
        for p_cD in p_cD_list:
            p_cE = p_cAtilde - p_cC - p_cD
            for p_cA in p_cA_list:
                if (p_cA-p_cC) > -1e-14 and (p_cC - p_cD)>-1e-14 and (p_cD - p_cE)>-1e-14 and p_cE>=-1e-14:
                    
                    committed_fraction = np.array([p_cA, 0, p_cC, p_cD, p_cE])
                    single_fraction = np.array([0, 1-sum(committed_fraction), 0, 0, 0])
                    committed_fraction_list.append(committed_fraction)
                    single_fraction_list.append(single_fraction)
    parallel_attractors(number_opinion, committed_fraction_list, single_fraction_list, des_file)
    return None 

def reduce_state_four(number_opinion):
    """TODO: Docstring for reduce_state.

    :number_opinion: TODO
    :returns: TODO

    """
    state = []
    for length in range(1, number_opinion+1):
        if length == 1 and number_opinion > 3:
            state.extend(['A', 'B', 'C', '~C', 'a','c', '~c' ])
        elif length == 2 and number_opinion == 4:
            state.extend(['AB', 'AC', 'A~C', 'BC', 'B~C','C~C'])
        elif length == 2 and number_opinion >= 5:
            state.extend(['AB', 'AC', 'A~C', 'BC', 'B~C','C~C','~C~C' ])

        elif length >2 and length < number_opinion-2:
            state.extend([i+ '~C' * (length-3) for i in ['ABC', 'AB~C', 'AC~C', 'BC~C', 'A~C~C', 'B~C~C', 'C~C~C', '~C~C~C']])
        elif length >2 and length == number_opinion-2:
            state.extend([i+ '~C' * (length-3) for i in ['ABC', 'AB~C', 'AC~C', 'BC~C', 'A~C~C', 'B~C~C', 'C~C~C']])
        elif length >2 and length == number_opinion-1:
            state.extend([i+ '~C' * (length-3) for i in ['ABC', 'AB~C', 'AC~C', 'BC~C']])
        elif length >2 and length == number_opinion:
            state.extend(['ABC' + '~C' * (length-3)])
    return state

def c_approximation_four(number_opinion):
    """TODO: Docstring for change_rule.

    :number_opinion: TODO
    :returns: TODO

    """
    possible_state = all_state(number_opinion)
    num_all_state = len(possible_state)
    C_tilde = possible_state[3:number_opinion]
    reduced = reduce_state_four(number_opinion)
    index_list = []
    for x in possible_state:
        if x != 'b':
            for single in C_tilde:
                x = x.replace(single, '~C').replace(single.lower(), '~c')
            index_list.append(reduced.index(x))
        elif x == 'b':
            index_list.append(-1)
    num_reduce_state = len(reduced)
    transition_before_list = []
    transition_after_list = []
    for s1 in possible_state:
        for s2 in possible_state:
            transition_before_list.append([s1, s2])
            transition_after_list.append(transition_rule(s1, s2))
    interaction_num = len(transition_after_list)
    change_matrix = np.zeros((interaction_num, num_reduce_state))
    for i in range(interaction_num):
        transition_after = transition_after_list[i]
        transition_before = transition_before_list[i]
        len_result = len(transition_after)
        for x in transition_before:
            if x !='b':
                for single in C_tilde:
                    x = x.replace(single, '~C').replace(single.lower(), '~c')
                #index = approximation_index(number_opinion, x_approx)
                index = reduced.index(x)
                change_matrix[i, index] -= 1
            
        for one_result in transition_after:
            for x in one_result:
                if x !='b':
                    for single in C_tilde:
                        x = x.replace(single, '~C').replace(single.lower(), '~c')
                    #index = approximation_index(number_opinion, x_approx)
                    index = reduced.index(x)
                    change_matrix[i, index] += 1/len_result
    c = change_matrix.transpose().reshape(num_reduce_state, num_all_state, num_all_state)
    c_approx1 = np.zeros((num_reduce_state, num_reduce_state, num_all_state))
    for i in range(num_reduce_state):
        index_combine = np.where(np.array(index_list) == i)[0]
        c_approx1[:, i, :] = np.average(c[:, index_combine, :], weights=[1] * len(index_combine), axis=1)

    c_approx2 = np.zeros((num_reduce_state, num_reduce_state, num_reduce_state))
    for i in range(num_reduce_state):
        index_combine = np.where(np.array(index_list) == i)[0]
        c_approx2[:, :, i] = np.average(c_approx1[:, :, index_combine], weights=[1] * len(index_combine), axis=2)
    return c_approx2

def attractors_approximation_four(number_opinion, committed_fraction, length, coefficient, des_file):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    t1 = time.time()
    initial_state = np.zeros((length))
    initial_state[4:7] = committed_fraction  # a, c, ~c 
    initial_state[1] = 1- sum(committed_fraction)
    t = np.arange(0, 1000, 0.01)
    #result = odeint(mf_ode, initial_state, t, args=(length, coefficient))[-1, :4]
    result = ode_stable_approximation(number_opinion, initial_state, length, coefficient)[-1, :4]
    data = np.hstack((committed_fraction, result))
    df_data = pd.DataFrame(data.reshape(1, len(data)))
    df_data.to_csv(des_file, index=None, header=None, mode='a')
    t2 = time.time()
    #print(committed_fraction, t2-t1)
    return None

def parallel_attractors_approximation_four(number_opinion_list, committed_fraction_list, des_file):
    """TODO: Docstring for parallel_attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    number_opinion_unique = np.unique(number_opinion_list)
    coefficient_list = []
    length_list = []
    for number_opinion in number_opinion_unique:
        coefficient = c_approximation_four(number_opinion)
        length = np.size(coefficient, 0)
        coefficient_list.append(coefficient)
        length_list.append(length)
    p = mp.Pool(cpu_number)
    p.starmap_async(attractors_approximation_four, [(number_opinion, committed_fraction, length_list[np.where(number_opinion == number_opinion_unique)[0][0]], coefficient_list[np.where(number_opinion == number_opinion_unique)[0][0]], des_file) for number_opinion, committed_fraction in zip(number_opinion_list, committed_fraction_list)]).get()
    p.close()
    p.join()
    return None

def approximation_oneuncommitted_four(number_opinion, p_list):
    """reduce some variables, for three-opinion variant, there are 6 variables: A, A', pA, AA', A'A', AA'A' 

    :number_opinion: TODO
    :: TODO
    :returns: TODO

    """
    des = f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_four/'
    if not os.path.exists(des):
        os.makedirs(des)
    for p in p_list:
        committed_fraction_list = []
        number_opinion_list = []
        p_cAtilde = (number_opinion - 2) * p
        p_A_list = np.round(np.arange(p+0.001, 0.1, 0.001), 10)
        des_file = des + f'p0={p}.csv'
        for p_A in p_A_list:
            p1 = p_A - 1e-3
            n1 = int(np.floor(p_cAtilde/p1))
            p_cCtilde = p1 * n1
            p_cC = p_cAtilde - p_cCtilde
            committed_fraction = np.array([p_A, p_cC, p_cCtilde])
            committed_fraction_list.append(committed_fraction)
            number_opinion_list.append(max(n1, 1) + 3)
        parallel_attractors_approximation_four(number_opinion_list, committed_fraction_list, des_file)
    return None

def approximation_oneuncommitted_three(number_opinion, p_list):
    """reduce some variables, for three-opinion variant, there are 6 variables: A, A', pA, AA', A'A', AA'A' 

    :number_opinion: TODO
    :: TODO
    :returns: TODO

    """
    des = f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_three/'
    for p in p_list:
        committed_fraction_list = []
        if not os.path.exists(des):
            os.makedirs(des)
        des_file = des+ f'p0={p}.csv'
        p_A_list = np.arange(p + 2e-4, 0.1, 0.001)
        for p_A in p_A_list:
            committed_fraction_list.append([p_A, (number_opinion-2) * p])
        parallel_attractors_approximation(number_opinion, committed_fraction_list, des_file)
    return None

def compare_original_reduced(number_opinion, committed_fraction):
    """TODO: Docstring for compare_original_reduced.

    :N: TODO
    :pA: TODO
    :p0: TODO
    :returns: TODO

    """
    start = 0
    end = 1000
    p_cA = committed_fraction[0]
    p_cAtilde = sum(committed_fraction[2:])
    t = np.arange(start, end, 0.01)
    length = 2**number_opinion -1 + number_opinion
    coefficient = change_rule(number_opinion)
    c_matrix = np.round(coefficient.reshape(length, length, length).transpose(2, 0, 1) , 15)
    mixed_fraction = np.zeros(( length-2*number_opinion))
    single_fraction = np.hstack((0, 1-sum(committed_fraction), np.zeros((number_opinion-2))))
    initial_state = np.hstack(([single_fraction, committed_fraction, mixed_fraction]))
    result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))

    coefficient_three = c_approximation(number_opinion)
    length_three = np.size(coefficient_three, 0)
    initial_three = np.zeros((length_three))
    initial_three[3:5] = np.array([committed_fraction[0], sum(committed_fraction[2:])])
    initial_three[1] = 1- sum(committed_fraction)
    result_three = odeint(mf_ode, initial_three, t, args=(length_three, coefficient_three))

    "S2"
    p_cAtilde = np.sum(committed_fraction[2:])
    p_cA = committed_fraction[0]
    n_max = int(np.floor(p_cAtilde/p_cA))
    p_cCtilde = (p_cA-1e-5) * n_max
    p_cC = p_cAtilde - p_cCtilde
    N = 3+ max(n_max, 1)

    coefficient_four = c_approximation_four(N)
    length_four = np.size(coefficient_four, 0)
    initial_four = np.zeros((length_four))
    initial_four[4:7] = np.array([p_cA, p_cC, p_cCtilde])
    initial_four[1] = 1- sum(committed_fraction)
    result_four = odeint(mf_ode, initial_four, t, args=(length_four, coefficient_four))

    plt.plot(t, result_three[:, 0], '-', color='tab:red', alpha=alpha, linewidth=lw, label='S1')
    plt.plot(t, result_four[:, 0], '-', color='tab:blue', alpha=alpha, linewidth=lw, label='S2')
    plt.xlabel('$t$', fontsize=fontsize)
    plt.plot(t, result[:, 0], '-', color='tab:green', alpha=alpha, linewidth=lw, label='original')
    plt.ylabel('$x_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=4.0)
    plt.locator_params(nbins=6)
    plt.show()
    return None

def small_committed(number_opinion, committed_fraction, single_fraction):
    """TODO: Docstring for one_realization.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    start = 0
    end = 100
    dt = 0.01
    length = 2**number_opinion -1 + number_opinion
    coefficient = change_rule(number_opinion, 1)
    c_matrix = np.round(coefficient.reshape(length, length, length).transpose(2, 0, 1) , 15)
    mixed_fraction = np.zeros(( length-2*number_opinion))
    t = np.arange(start, end, dt)
    initial_state = np.hstack(([single_fraction, committed_fraction, mixed_fraction]))
    result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))
    single_result = result[:, :number_opinion]
    mix_result = result[:, 2 * number_opinion:]
    normal_single = single_result[:, 2:]/committed_fraction[2:] ** 2
    normal_A = single_result[:, 0]/committed_fraction[0] ** 2

    """
    "single state x--t"
    #plt.plot(t, mix_result[:, 0], linewidth=lw, alpha=alpha, label='AB')
    plt.plot(t, single_result[:, 2], linewidth=lw, alpha=alpha, label='C')
    plt.plot(t, single_result[:, 3], linewidth=lw, alpha=alpha, label='D')
    plt.plot(t, single_result[:, 4], linewidth=lw, alpha=alpha, label='E')
    plt.plot(t, single_result[:, 5], linewidth=lw, alpha=alpha, label='F')
    plt.plot(t, single_result[:, 0], linewidth=lw, alpha=alpha, label='A')
    plt.plot(t, single_result[:, 1], linewidth=lw, alpha=alpha, label='B')
    plt.ylabel('$x$', fontsize=fontsize)
    plt.xlabel('$t$', fontsize=fontsize)

    """
    "single state x/P^2--t"
    #plt.plot(t, single_result[:, 1]/committed_fraction[0] **2, linewidth=lw, alpha=alpha, label='B')
    plt.plot(t, normal_single[:, 0], linewidth=lw, alpha=alpha, label='C')
    plt.plot(t, normal_single[:, 1], linewidth=lw, alpha=alpha, label='D')
    plt.plot(t, normal_single[:, 2], linewidth=lw, alpha=alpha, label='E')
    plt.plot(t, normal_single[:, 3], linewidth=lw, alpha=alpha, label='F')
    plt.plot(t, single_result[:, 0]/committed_fraction[0] **2, linewidth=lw, alpha=alpha, label='A')
    plt.ylabel('$x_i/P_i^2$', fontsize=fontsize)
    plt.xlabel('$t$', fontsize=fontsize)

    """
    "x_i/P_i^2--x_C/P_C^2"
    plt.plot(normal_single[:, 0], normal_single[:, 1], linewidth=lw, alpha=alpha, label='D')
    plt.plot(normal_single[:, 0], normal_single[:, 2], linewidth=lw, alpha=alpha, label='E')
    plt.plot(normal_single[:, 0], normal_single[:, 3], linewidth=lw, alpha=alpha, label='F')
    plt.plot(normal_single[:, 0], normal_single[:, 0], '--', linewidth=lw, alpha=alpha, color='grey', label='C')
    plt.ylabel('$x_i/P_i^2$', fontsize=fontsize)
    plt.xlabel('$x_C/P_C^2$', fontsize=fontsize)
    """

    """
    "x_Ai--t"
    plt.plot(t, mix_result[:, 1], linewidth=lw, alpha=alpha, label='AC')
    plt.plot(t, mix_result[:, 2], linewidth=lw, alpha=alpha, label='AD')
    plt.plot(t, mix_result[:, 3], linewidth=lw, alpha=alpha, label='AE')
    plt.plot(t, mix_result[:, 4], linewidth=lw, alpha=alpha, label='AF')
    plt.ylabel('$x$', fontsize=fontsize)
    plt.xlabel('$t$', fontsize=fontsize)
    """

    """
    "x_Ai--t"
    plt.plot(t, mix_result[:, 1], linewidth=lw, alpha=alpha, label='AC')
    plt.plot(t, mix_result[:, 2], linewidth=lw, alpha=alpha, label='AD')
    plt.plot(t, mix_result[:, 3], linewidth=lw, alpha=alpha, label='AE')
    plt.plot(t, mix_result[:, 4], linewidth=lw, alpha=alpha, label='AF')
    plt.ylabel('$x$', fontsize=fontsize)
    plt.xlabel('$t$', fontsize=fontsize)
    """

    """
    "x_Ai/P_i--x_AC/P_C"
    plt.plot(mix_result[:, 1]/committed_fraction[2], mix_result[:, 2]/committed_fraction[3], linewidth=lw, alpha=alpha, label='AD')
    plt.plot(mix_result[:, 2]/committed_fraction[3], mix_result[:, 3]/committed_fraction[4], linewidth=lw, alpha=alpha, label='AE')
    plt.plot(mix_result[:, 3]/committed_fraction[4], mix_result[:, 4]/committed_fraction[5], linewidth=lw, alpha=alpha, label='AF')
    plt.plot(mix_result[:, 1]/committed_fraction[2], mix_result[:, 1]/committed_fraction[2], '--', linewidth=lw, alpha=alpha, color='grey', label='AC')
    plt.ylabel('$x_{Ai}/P_i$', fontsize=fontsize)
    plt.xlabel('$x_{AC}/P_C$', fontsize=fontsize)
    """

    """
    "x_Ai/P_i--x_AC/P_C"
    plt.plot(t, mix_result[:, 1]/committed_fraction[2], linewidth=lw, alpha=alpha, label='AC')
    plt.plot(t, mix_result[:, 2]/committed_fraction[3], linewidth=lw, alpha=alpha, label='AD')
    plt.plot(t, mix_result[:, 3]/committed_fraction[4], linewidth=lw, alpha=alpha, label='AE')
    plt.plot(t, mix_result[:, 4]/committed_fraction[5], linewidth=lw, alpha=alpha, label='AF')
    plt.ylabel('$x_{Ai}/P_i$', fontsize=fontsize)
    plt.xlabel('$t$', fontsize=fontsize)
    """

    """
    "x_Bi--t"
    plt.plot(t, mix_result[:, 5], linewidth=lw, alpha=alpha, label='BC')
    plt.plot(t, mix_result[:, 6], linewidth=lw, alpha=alpha, label='BD')
    plt.plot(t, mix_result[:, 7], linewidth=lw, alpha=alpha, label='BE')
    plt.plot(t, mix_result[:, 8], linewidth=lw, alpha=alpha, label='BF')
    plt.ylabel('$x_{Bi}$', fontsize=fontsize)
    plt.xlabel('$t$', fontsize=fontsize)
    """

    """
    "x_Bi/P_i--t"
    plt.plot(t, mix_result[:, 5]/committed_fraction[2], linewidth=lw, alpha=alpha, label='BC')
    plt.plot(t, mix_result[:, 6]/committed_fraction[3], linewidth=lw, alpha=alpha, label='BD')
    plt.plot(t, mix_result[:, 7]/committed_fraction[4], linewidth=lw, alpha=alpha, label='BE')
    plt.plot(t, mix_result[:, 8]/committed_fraction[5], linewidth=lw, alpha=alpha, label='BF')
    plt.ylabel('$x_{Bi}/P_i$', fontsize=fontsize)
    plt.xlabel('$t$', fontsize=fontsize)
    """

    """
    "x_Bi/P_i--x_BC/P_C"
    plt.plot(mix_result[:, 5]/committed_fraction[2], mix_result[:, 6]/committed_fraction[3], linewidth=lw, alpha=alpha, label='BD')
    plt.plot(mix_result[:, 5]/committed_fraction[2], mix_result[:, 7]/committed_fraction[4], linewidth=lw, alpha=alpha, label='BE')
    plt.plot(mix_result[:, 5]/committed_fraction[2], mix_result[:, 8]/committed_fraction[5], linewidth=lw, alpha=alpha, label='BF')
    plt.plot(mix_result[:, 5]/committed_fraction[2], mix_result[:, 5]/committed_fraction[2], '--', linewidth=lw, alpha=alpha, color='grey', label='BC')
    plt.ylabel('$x_{Bi}/P_i$', fontsize=fontsize)
    plt.xlabel('$x_{BC}/P_C$', fontsize=fontsize)
    """

    """
    "x_ABi--t"
    plt.plot(t, mix_result[:, 15], linewidth=lw, alpha=alpha, label='ABC')
    plt.plot(t, mix_result[:, 16], linewidth=lw, alpha=alpha, label='ABD')
    plt.plot(t, mix_result[:, 17], linewidth=lw, alpha=alpha, label='ABE')
    plt.plot(t, mix_result[:, 18], linewidth=lw, alpha=alpha, label='ABF')
    plt.ylabel('$x_{ABi}$', fontsize=fontsize)
    plt.xlabel('$t$', fontsize=fontsize)
    """

    """
    "x_ABi/P_i--t"
    plt.plot(t, mix_result[:, 15]/committed_fraction[2], linewidth=lw, alpha=alpha, label='ABC')
    plt.plot(t, mix_result[:, 16]/committed_fraction[3], linewidth=lw, alpha=alpha, label='ABD')
    plt.plot(t, mix_result[:, 17]/committed_fraction[4], linewidth=lw, alpha=alpha, label='ABE')
    plt.plot(t, mix_result[:, 18]/committed_fraction[5], linewidth=lw, alpha=alpha, label='ABF')
    plt.ylabel('$x_{ABi}/P_i$', fontsize=fontsize)
    plt.xlabel('$t$', fontsize=fontsize)
    """
    
    """
    "x_ABi/P_i--x_ABC_P_C"
    plt.plot(mix_result[:, 15]/committed_fraction[2], mix_result[:, 16]/committed_fraction[3], linewidth=lw, alpha=alpha, label='ABD')
    plt.plot(mix_result[:, 15]/committed_fraction[2], mix_result[:, 17]/committed_fraction[4], linewidth=lw, alpha=alpha, label='ABE')
    plt.plot(mix_result[:, 15]/committed_fraction[2], mix_result[:, 18]/committed_fraction[5], linewidth=lw, alpha=alpha, label='ABF')
    plt.plot(mix_result[:, 15]/committed_fraction[2], mix_result[:, 15]/committed_fraction[2], '--', linewidth=lw, alpha=alpha, color='grey', label='ABC')
    plt.ylabel('$x_{ABi}/P_i$', fontsize=fontsize)
    plt.xlabel('$x_{ABC}/P_C$', fontsize=fontsize)
    """

    plt.subplots_adjust(left=0.20, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.show()

    return result

def single_peak_A(number_opinion, pA_list, gamma_list):
    """TODO: Docstring for single_peak_A.

    :number_op: TODO
    :returns: TODO

    """
    dt = 0.001
    t = np.arange(0, 100, dt)
    length = 4 * number_opinion - 3
    c_matrix = change_rule_approximation_three(number_opinion)

    des = f'../data/num_peak/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'number_opinion={number_opinion}.csv' 
    for pA in pA_list:
        for gamma in gamma_list:
            p0 = gamma * pA
            pA_tilde = p0 * (number_opinion-2)
            initial_state = np.zeros((length))
            committed_fraction = np.array([pA, pA_tilde])
            initial_state[3:5] = committed_fraction  
            initial_state[1] = 1- sum(committed_fraction)
            if sum(committed_fraction) < 1:
                result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))
                dxA = np.diff(result[:, 0])
                dxA_max = np.max(dxA)
                peak_num = 0
                index = argrelextrema(dxA, np.greater)[0]
                for i in index:
                    if dxA[i] > 0.01 * dxA_max:
                        if np.all(np.diff(dxA[i-10:i])>0) and np.all(np.diff(dxA[i:i+10])<0):
                            peak_num += 1

                #xA = xA[:np.argmax(xA) + 2]
                #second_peak = np.heaviside(np.sum(np.diff(xA[np.where(np.diff(xA)<0)[0][0]:])>0), 0)
                data = np.hstack((pA, gamma, peak_num))
                df_data = pd.DataFrame(data.reshape(1, len(data)))
                #df_data.to_csv(des_file, index=None, header=None, mode='a')
                print(np.max(result[-1, 1]))
                plt.plot(t[1:], np.diff(result[:, 0])/dt, linewidth=2, alpha=alpha, label='A', color='#66c2a5')
                plt.plot(t[1:], np.diff(result[:, 2])/dt, linewidth=2, alpha=alpha, label='$\\tilde{A}$', color='#fc8d62')
                #plt.plot(result[:, 2])
                plt.xlabel('$t$', fontsize=fontsize)
                plt.ylabel('$dx/dt$', fontsize=fontsize)
                plt.xticks(fontsize=ticksize)
                plt.yticks(fontsize=ticksize)
                plt.subplots_adjust(left=0.24, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
                plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
                plt.locator_params(nbins=6)
                plt.show()

                plt.plot(t, result[:, 0], linewidth=2, alpha=alpha, label='A', color='#66c2a5')
                plt.plot(t, result[:, 2], linewidth=2, alpha=alpha, label='$\\tilde{A}$', color='#fc8d62')
                #plt.plot(result[:, 2])
                plt.xlabel('$t$', fontsize=fontsize)
                plt.ylabel('$x$', fontsize=fontsize)
                plt.xticks(fontsize=ticksize)
                plt.yticks(fontsize=ticksize)
                plt.subplots_adjust(left=0.24, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
                plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
                plt.locator_params(nbins=6)

                plt.show()
    return None

def small_committed_approximation(number_opinion, pA_list, p_list):
    """TODO: Docstring for one_realization.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    des = f'../data/approximation_compare/number_opinion={number_opinion}/'
    if not os.path.exists(des):
        os.makedirs(des)
    coefficient = change_rule_approximation_three(number_opinion)
    length = np.size(coefficient, 0)
    for pA in pA_list:
        for p in p_list:
            des_file = des+ f'pA={pA}_p={p}.csv'
            initial_state = np.zeros((length))
            pAtilde = p * (number_opinion - 2)
            committed_fraction = np.array([pA, pAtilde])
            initial_state[3:5] = committed_fraction  
            initial_state[1] = 1- sum(committed_fraction)
            dt = 0.01
            t = np.arange(0, 500, dt)
            result = ode_stable_approximation(number_opinion, initial_state, length, coefficient)[::int(1/dt), :3]
            df_data = pd.DataFrame(result)
            df_data.to_csv(des_file, index=None, header=None, mode='a')
    return result

    
def mix_sending_single(number_opinion):
    """TODO: Docstring for mix_sending_single.

    :number_op: TODO
    :returns: TODO

    """
    possible_state = all_state(number_opinion)
    length = 2**number_opinion -1 + number_opinion
    mix_single = np.zeros((number_opinion, length - 2*number_opinion))
    for index, i in enumerate(possible_state[2*number_opinion:]):
        for j in i:
            mix_single[possible_state.index(j), index] =  1/len(i)
    return mix_single

def mix_sending_single_approximation(number_opinion):
    """TODO: Docstring for mix_sending_single.

    :number_op: TODO
    :returns: TODO

    """
    possible_state = all_state_approximation_three(number_opinion)
    length = len(possible_state)
    mix_single = np.zeros((3, length - 5))
    for index, i in enumerate(possible_state[5:]):
        for j in i:
            mix_single[possible_state.index(j), index] += 1/len(i)
    return mix_single




def entropy(number_opinion, committed_fraction, single_fraction):
    """TODO: Docstring for entropy.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    start = 0
    end = 100
    dt = 0.01
    length = 2**number_opinion -1 + number_opinion
    coefficient = change_rule(number_opinion)
    c_matrix = np.round(coefficient.reshape(length, length, length).transpose(2, 0, 1) , 15)
    mixed_fraction = np.zeros(( length-2*number_opinion))
    t = np.arange(start, end, dt)
    initial_state = np.hstack(([single_fraction, committed_fraction, mixed_fraction]))
    result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))
    single_all = (result[:, :number_opinion] + result[:, number_opinion:2*number_opinion])
    mix_single = mix_sending_single(number_opinion)
    mix_all = []
    for one_mix_single in mix_single:
        mix_all.append(np.sum(result[:, 2*number_opinion:] * one_mix_single, 1))
    ps = single_all.transpose() /np.sum(single_all, 1)
    ps = single_all.transpose() + np.vstack(( mix_all))
    pslogps =  ps * np.log(ps)
    nan_index = np.where(ps <=0)
    pslogps[nan_index] = 0
    H = - np.sum(pslogps, 0)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(t, result[:, 0], linewidth=3, alpha=alpha, label='$A$')
    ax1.plot(t, result[:, 1], linewidth=3, alpha=alpha, label='$B$')
    ax1.plot(t, result[:, 2], linewidth=3, alpha=alpha, label='$C$')
    ax2.plot(t, H, '--', linewidth=3, alpha=alpha, color='tab:red') 
    ax1.set_xlabel('t', fontsize=fontsize)
    ax1.set_ylabel('x', fontsize=fontsize)
    ax2.set_ylabel('$\\mathcal{H}$', fontsize=fontsize, color='tab:red')
    ax1.tick_params(axis='x', labelsize=ticksize) 
    ax1.tick_params(axis='y', labelsize=ticksize) 
    ax2.tick_params(axis='y',labelsize=ticksize, colors='tab:red') 
    plt.subplots_adjust(left=0.18, right=0.85, wspace=0.25, hspace=0.25, bottom=0.15, top=0.90)
    ax1.legend(frameon=False, fontsize = legendsize)
    plt.locator_params(nbins=6)
    #plt.savefig(save_des)
    #plt.close('all')

    return ps

def entropy_approximation(number_opinion, pA, p):
    """TODO: Docstring for entropy.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    start = 0
    end = 100
    dt = 0.01
    t = np.arange(start, end, dt)
    coefficient = change_rule_approximation_three(number_opinion)
    length = np.size(coefficient, 0)
    initial_state = np.zeros((length))
    pAtilde = p * (number_opinion - 2)
    committed_fraction = np.array([pA, pAtilde])
    initial_state[3:5] = committed_fraction  
    initial_state[1] = 1- sum(committed_fraction)
    result = odeint(mf_ode, initial_state, t, args=(length, coefficient))
    single_all = result[:, :3].copy()
    single_all[:, 0] = single_all[:, 0] + result[:, 3]
    single_all[:, 2] = single_all[:, 2] + result[:, 4]
    mix_single = mix_sending_single_approximation(number_opinion)
    mix_all = []
    for one_mix_single in mix_single:
        mix_all.append(np.sum(result[:, 5:] * one_mix_single, 1))
    ps = single_all.transpose() /np.sum(single_all, 1)
    ps = single_all.transpose() + np.vstack((mix_all))
    ps_split = np.zeros((number_opinion, len(t)))
    ps_split[:2] = ps[:2].copy()
    ps_split[2:] = ps[-1]/(number_opinion - 2)
    
    pslogps =  ps_split * np.log(ps_split)
    nan_index = np.where(ps_split <=0)
    pslogps[nan_index] = 0
    H = - np.sum(pslogps, 0)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.plot(t, result[:, 0], linewidth=3, alpha=alpha, label='$A$')
    ax1.plot(t, result[:, 1], linewidth=3, alpha=alpha, label='$B$')
    ax1.plot(t, result[:, 2], linewidth=3, alpha=alpha, label='$C$')
    ax2.plot(t, H, '--', linewidth=3, alpha=alpha, color='tab:red') 
    ax1.set_xlabel('t', fontsize=fontsize)
    ax1.set_ylabel('x', fontsize=fontsize)
    ax2.set_ylabel('$\\mathcal{H}$', fontsize=fontsize, color='tab:red')
    ax1.tick_params(axis='x', labelsize=ticksize) 
    ax1.tick_params(axis='y', labelsize=ticksize) 
    ax2.tick_params(axis='y',labelsize=ticksize, colors='tab:red') 
    plt.subplots_adjust(left=0.18, right=0.85, wspace=0.25, hspace=0.25, bottom=0.15, top=0.90)
    ax1.legend(frameon=False, fontsize = legendsize)
    plt.locator_params(nbins=6)
    #plt.savefig(save_des)
    #plt.close('all')

    return ps

def entropy_consensus(number_opinion, pA_list, p_list):
    """TODO: Docstring for entropy_consensus.

    :number_opoinion: TODO
    :pA_list: TODO
    :p_list: TODO
    :returns: TODO

    """
    for p in p_list:
        H_list = []
        consensus_t = []
        for pA in pA_list:
            des_mfe = f'../data/approximation_compare/number_opinion={number_opinion}/pA={pA}_p={p}.csv'
            data_mfe = np.array(pd.read_csv(des_mfe, header=None))
            xA_mfe = data_mfe[:, 0]
            xB_mfe = data_mfe[:, 1]
            xC_mfe = data_mfe[:, 2]
            t = np.where(xA_mfe > 0.9 * xA_mfe[-1])[0][0]
            consensus_t.append(t)
            pAtilde = p * (number_opinion - 2)
            ps = np.array([pA, 1-pA-pAtilde, pAtilde])
            ps = np.hstack((np.array([pA, 1-pA-pAtilde]), np.ones(number_opinion - 2) * p))
            H = - np.sum(ps * np.log(ps))
            H_list.append(H)
            #plt.plot(xA_mfe)

        plt.plot(H_list, consensus_t, 'o-', linewidth=lw, alpha=alpha, label=f'$p_0={p}$')    
    plt.xlabel('entropy $\\mathcal{H}$', fontsize=fontsize)
    plt.ylabel('consensus time $T$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.locator_params(nbins=6)

    #plt.show()
    return H_list





number_opinion = 12
committed_fraction = np.array([0.2, 0.0, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
single_fraction = np.array([0, 1-sum(committed_fraction), 0, 0, 0, 0 , 0, 0, 0, 0])
#ps = entropy(number_opinion, committed_fraction, single_fraction)
pA = 0.2
p =0.01

#entropy_approximation(number_opinion, pA, p)
pA_list = np.array([0.1, 0.12, 0.14, 0.16, 0.18, 0.2])
pA_list = np.array([0.14])
p_list = np.array([0.01, 0.02, 0.03, 0.04])
H = entropy_consensus(number_opinion, pA_list, p_list)
