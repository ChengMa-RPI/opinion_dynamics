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

cpu_number = 40
fontsize = 22
ticksize= 15
legendsize = 16
alpha = 0.8
lw = 3


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

def c_approximation_three(number_opinion):
    """TODO: Docstring for change_rule.

    :number_opinion: TODO
    :returns: TODO

    """
    possible_state = all_state(number_opinion)
    reduce_state =['A', 'A~', 'a', 'AA~', 'A~A~', 'AA~A~']
    transition_before_list = []
    transition_after_list = []
    for s1 in possible_state:
        for s2 in possible_state:
            transition_before_list.append([s1, s2])
            transition_after_list.append(transition_rule(s1, s2))
    interaction_num = len(transition_after_list)
    change_matrix = np.zeros((interaction_num, len(reduce_state)))
    for i in range(interaction_num):
        transition_after = transition_after_list[i]
        transition_before = transition_before_list[i]
        len_result = len(transition_after)
        for x in transition_before:
            if x !='b' and x != 'c':

                x_approx = x.replace('B', 'A~').replace('C', 'A~')
                index = reduce_state.index(x_approx)
                change_matrix[i, index] -= 1
            
        for one_result in transition_after:
            for x in one_result:
                if x !='b' and x != 'c':
                    x_approx = x.replace('B', 'A~').replace('C', 'A~')
                    index = reduce_state.index(x_approx)
                    change_matrix[i, index] += 1/len_result
    c = change_matrix.transpose().reshape(6, 10, 10)
    c_approx1 = np.zeros((6, 6, 10))
    c_approx1[:, 0, :] = c[:, 0, :]
    c_approx1[:, 1, :] = np.average(c[:, 1:3, :], weights=[1,1], axis=1)
    c_approx1[:, 2, :] = c[:, 3, :]
    c_approx1[:, 3, :] = np.average(c[:, 6:8, :], weights=[1,1], axis=1)
    c_approx1[:, -2:, :] = c[:, -2:, :]

    c_approx2 = np.zeros((6, 6, 6))
    c_approx2[:, :, 0] = c_approx1[:, :, 0]
    c_approx2[:, :, 1] = np.average(c_approx1[:, :, 1:3], weights=[1,1], axis=2)
    c_approx2[:, :, 2] = c_approx1[:, :, 3]
    c_approx2[:, :, 3] = np.average(c_approx1[:, :, 6:8], weights=[1,1], axis=2)
    c_approx2[:, :, -2:] = c_approx1[:, :, -2:]

    return c_approx2

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

def ode_stable_loop(number_opinion, committed_fraction, single_fraction, c_matrix):
    """TODO: Docstring for ode_stable.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    start = 0
    end = 100
    difference = 1
    single_fraction1 = single_fraction
    length = 2**number_opinion -1 + number_opinion
    mixed_fraction = np.zeros(( length-2*number_opinion))
    while abs(difference) > 1e-12:
        t = np.arange(start, end, 0.01)
        initial_state = np.hstack(([single_fraction1, committed_fraction, mixed_fraction]))
        result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))
        single_fraction2 = result[-1, :number_opinion]
        mixed_fraction = result[-1, 2*number_opinion:]
        difference = sum(abs(single_fraction2 - single_fraction1))
        single_fraction1 = single_fraction2
    return single_fraction1

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
    
def bifurcation(number_opinion):
    """TODO: Docstring for bifurcation.

    :number_opinion: TODO
    :returns: TODO

    """
    region1 = []
    if number_opinion == 2:
        for p in np.arange(0, 0.3, 0.01):
            for q in np.arange(0, 0.3, 0.01):
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

def attractors_three(number_opinion, committed_fraction, des_file):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """

    t1 = time.time()
    attractor_list = []
    uncommitted_fraction = np.round(1 - sum(committed_fraction), digit)
    length = 2**number_opinion -1 + number_opinion
    coefficient = change_rule(number_opinion)
    c_matrix = np.round(coefficient.reshape(length, length, length).transpose(2, 0, 1) , 15)
    for a in np.arange(0, uncommitted_fraction+1e-6, 0.01):
        for b in np.arange(0, uncommitted_fraction-a+1e-6, 0.01):
            c = uncommitted_fraction-a-b
            single_fraction = np.round(np.array([a, b, c]), digit)
            attractor = ode_stable(number_opinion, committed_fraction, single_fraction, c_matrix)
            print(single_fraction, attractor)
            attractor_list.append(attractor)
    attractor_unique = np.unique(np.round(np.vstack((attractor_list)), 4), axis=0)
    data = np.hstack((committed_fraction, np.ravel(attractor_unique)))
    df_data = pd.DataFrame(data.reshape(1, len(data)))
    #df_data.to_csv(des_file, index=None, header=None, mode='a')

    t2 = time.time()
    print(committed_fraction, t2-t1, attractor_unique)

    return None

def attractors_four(number_opinion, committed_fraction, des_file):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """

    t1 = time.time()
    attractor_list = []
    uncommitted_fraction = np.round(1 - sum(committed_fraction), digit)
    length = 2**number_opinion -1 + number_opinion
    coefficient = change_rule(number_opinion)
    c_matrix = np.round(coefficient.reshape(length, length, length).transpose(2, 0, 1), 15)
    single_fraction = np.round(np.array([0, 0, 0, uncommitted_fraction]), digit)
    attractor = ode_stable(number_opinion, committed_fraction, single_fraction, c_matrix)
    print(single_fraction, attractor)
    attractor_list.append(attractor)
    attractor_unique = np.unique(np.round(np.vstack((attractor_list)), 4), axis=0)
    data = np.hstack((committed_fraction, np.ravel(attractor_unique)))
    df_data = pd.DataFrame(data.reshape(1, len(data)))
    df_data.to_csv(des_file, index=None, header=None, mode='a')
    t2 = time.time()
    print(committed_fraction, t2-t1, attractor_unique)

    return data

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

def parallel_attractors_full(number_opinion, committed_fraction_list):
    """TODO: Docstring for parallel_attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    des = '../data/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'num_opinion={number_opinion}_stable.csv'
    p = mp.Pool(cpu_number)
    if number_opinion == 3:
        function = attractors_three
    elif number_opinion == 4:
        function = attractors_four
    p.starmap_async(function, [(number_opinion, committed_fraction, des_file) for committed_fraction in committed_fraction_list]).get()
    p.close()
    p.join()

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
    end = 1000
    length = 2**number_opinion -1 + number_opinion
    coefficient = change_rule(number_opinion)
    c_matrix = np.round(coefficient.reshape(length, length, length).transpose(2, 0, 1) , 15)
    mixed_fraction = np.zeros(( length-2*number_opinion))
    t = np.arange(start, end, 0.01)
    initial_state = np.hstack(([single_fraction, committed_fraction, mixed_fraction]))
    result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))
    plt.plot(t, result[:, 0], '-', alpha=alpha, linewidth=lw, label='A')
    plt.plot(t, result[:, 1], '-.', alpha=alpha, linewidth=lw, label='B')
    plt.plot(t, result[:, 2], '--', alpha=alpha, linewidth=lw, label='C')
    plt.plot(t, result[:, 3], '--', alpha=alpha, linewidth=lw, label='D')
    plt.xlabel('$t$', fontsize=fontsize)
    plt.ylabel('$x$', fontsize=fontsize)
    plt.subplots_adjust(left=0.15, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
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
    #result = odeint(mf_ode, initial_state, t, args=(length, coefficient))[-1]
    result = odeint(mf_ode, initial_state, t, args=(length, coefficient))[-1, :3]
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
    coefficient = c_approximation(number_opinion)
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

def polarization():
    """TODO: Docstring for plorization.

    :arg1: TODO
    :returns: TODO

    """
    number_opinion = 4
    p_cA_list = np.arange(0, 0.1, 0.0001)
    p_cAtilde = np.arange(0.01, 0.1, 0.01)
    for p in p_cAtilde:
        p = round(p, 2)
        print(p)
        committed_fraction_list = []
        single_fraction_list = []
        des_file = f'../data/polarization/P_cAtilde={p}.csv'
        for p_cA in p_cA_list:
            p_max_list = np.arange(p/2, p + 0.001, 0.01)
            for p_max in p_max_list:
                p_min = p - p_max
                committed_fraction = np.array([p_cA, 0, p_max, p_min])
                single_fraction = np.array([0, 1-sum(committed_fraction), 0, 0])
                committed_fraction_list.append(committed_fraction)
                single_fraction_list.append(single_fraction)
        parallel_attractors(number_opinion, committed_fraction_list, single_fraction_list, des_file)
        
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





number_opinion = 3
digit = 3

committed_fraction_list = []
for p in np.arange(0, 0.15, 0.01):
    for q in np.arange(0, min(1-p, 0.15), 0.01):
        for r in np.arange(0, min(1-p-q, 0.15), 0.01):
            committed_fraction = np.array([p, q, r])
            committed_fraction_list.append(committed_fraction)
committed_fraction_list = np.round(np.vstack((committed_fraction_list)), digit)


committed_fraction = np.round(np.array([0.1, 0.1, 0.05]), digit)
initial_single = np.array([0., 0.485-1e-6, 0.475+1e-6])
#one_realization(number_opinion, committed_fraction, initial_single)
#basin_attraction(number_opinion, committed_fraction)
t1 = time.time()
#attractor_list = attractors(number_opinion, committed_fraction, '../data')
t2 = time.time()

number_opinion = 4
committed_fraction_list = []
single_fraction_list = []

cA_list = np.arange(0, 0.2, 0.002)
cB_list = np.arange(0, 0.15, 0.015)
for cA in cA_list:
    for cB in cB_list:
        committed_fraction = np.round(np.hstack((cA, np.ones(number_opinion -2) * cB, 0)), 3)
        uC = 1 - sum(committed_fraction)
        single_fraction = np.round(np.hstack((0 * np.ones(number_opinion - 1), uC )), 3)
        committed_fraction_list.append(committed_fraction)
        single_fraction_list.append(single_fraction)
des_file = f'num_opinion={number_opinion}_oneuncommitted.csv'



""
number_opinion_list = [4]
pA_list = np.round(np.arange(0, 0.2, 0.0001), 4)
p_not_A_list = np.round(np.arange(0, 0.2, 0.001), 3)
for number_opinion in number_opinion_list:
    approximation_oneuncommitted(number_opinion, pA_list, p_not_A_list)
    pass


"Introduce fluctuations"
number_opinion = 5
cA_list = np.arange(0.06, 0.10, 0.0001)
p = 0.07
sigma= 0.005
seed_list = np.arange(100).tolist()
normalization = 1
sigma_list = [0.005]
p_list = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065]
p_list = [0.005]
for sigma in sigma_list:
    for p in p_list:
        #fluctuate_oneuncommitted(number_opinion, cA_list, p, sigma, seed_list, normalization)
        pass


for number_opinion in number_opinion_list:
    #approximation_oneuncommitted_two(number_opinion, pA_list)
    pass

number_opinion = 3
p_list = np.round(np.arange(0.005, 0.4, 0.005), 3)
for p in p_list:
    #fluctuate_lowerbound2(number_opinion, p)
    pass
p_list =  [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065]
p_list = [0.048]

for number_opinion in [6]:
    for p in p_list:
        #fluctuate_lowerbound(number_opinion, p)
        pass
