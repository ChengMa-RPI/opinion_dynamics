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

cpu_number = 6
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

def transition_rule_approximation_S3(s1, s2, pi):
    """states after interaction (speaker-listenser) 7 states: A, B, C, AB, AC, BC, ABC
        

    :s1: TODO
    :s2: TODO
    :returns: TODO

    """

    pi_square = np.sum(pi ** 2)/np.sum(pi)**2
    pi_cubic = np.sum(pi ** 3)/np.sum(pi**2)/np.sum(pi)
    pi_fourth = np.sum(pi ** 4)/np.sum(pi ** 2) ** 2
    """
    pi_square = 1/len(pi)
    pi_cubic = 1/len(pi)
    pi_fourth = 1/len(pi)
    """

    n1 = s1.count('C')
    n2 = s2.count('C')
    if n2 == 1:
        pi_square_n2 = n2 * np.sum(pi**2 * (1-pi) ** (n2-1))/np.sum(pi)**2
        pi_cubic_n2 = n2 * np.sum(pi**3 * (1-pi) ** (n2-1))/np.sum(pi**2)/np.sum(pi)
    else:
        pi_square_n2 = n2/len(pi)
        pi_cubic_n2 = n2/len(pi)


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
                if len(s2) == 1:
                    p_1 = pi_cubic_n2
                else: 
                    p_1 = pi_square_n2
                sf2 = s2 + v
                sf2_2 = ''.join(sorted(sf2))
                p_2= 1-p_1
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
                    if len(s1) == 1:
                        p_1 = pi_cubic
                    else:
                        p_1 = pi_square
                    sf1_2 = s1
                    p_2= 1-p_1
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
                    if len(s1) == 1 and len(s2) ==1:
                        p_1 =  pi_fourth * n2
                    elif len(s1) > 1 and len(s2) >1:
                        p_1 = pi_square_n2
                    else:
                        p_1 = pi_cubic_n2

                    sf1_2 = s1
                    sf2_2 = v + s2
                    sf2_2 = ''.join(sorted(sf2_2))
                    p_2 = 1-p_1
                    result.append([(sf1_1, sf2_1, p_1), (sf1_2, sf2_2, p_2)])
                else:
                    sf1 = s1
                    sf2 = v + s2
                    sf2 = ''.join(sorted(sf2))
                    result.append([(sf1, sf2, 1)])
    return result

def change_rule_approximation_S3(number_opinion, small_committed):
    """TODO: Docstring for change_rule.

    :number_opinion: TODO
    :returns: TODO

    """
    #possible_state = ['A', 'B', 'C', 'a', 'c', 'AB', 'AC', 'BC', 'ABC']
    possible_state = all_state_approximation_three(number_opinion)

    length = len(possible_state)
    pi = small_committed
    transition_before_list = []
    transition_after_list = []
    for s1 in possible_state:
        for s2 in possible_state:
            transition_before_list.append([s1, s2])
            transition_after_list.append(transition_rule_approximation_S3(s1, s2, small_committed))
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

def attractors_approximation_S3(number_opinion, committed_fraction, des_file):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    t1 = time.time()
    small_committed = committed_fraction[2:]
    length = 4 * number_opinion -3
    initial_state = np.zeros((length))
    initial_state[1] = 1- sum(committed_fraction)
    initial_state[3] = committed_fraction[0]
    initial_state[4] = sum(small_committed)

    t = np.arange(0, 500, 0.01)
    coefficient = change_rule_approximation_three(number_opinion, small_committed)
    result = ode_stable_approximation(number_opinion, initial_state, length, coefficient)[-1, :3]
    data = np.hstack((committed_fraction, result))
    df_data = pd.DataFrame(data.reshape(1, len(data)))
    df_data.to_csv(des_file, index=None, header=None, mode='a')
    t2 = time.time()
    #print(committed_fraction, t2-t1)
    return None

def parallel_attractors_approximation_S3(number_opinion, committed_fraction_list, des_file):
    """TODO: Docstring for parallel_attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(attractors_approximation_S3, [(number_opinion, committed_fraction, des_file) for committed_fraction in committed_fraction_list]).get()
    p.close()
    p.join()
    return None

def approximation_oneuncommitted(number_opinion, p, num_iter, seed=0):
    """reduce some variables, for three-opinion variant, there are 6 variables: A, A', pA, AA', A'A', AA'A' 

    :number_opinion: TODO
    :: TODO
    :returns: TODO

    """
    np.random.seed(seed)
    des = f'../data/num_opinion={number_opinion}_first_order/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'p={p}_random.csv'
    committed_fraction_list = []
    p_cAtilde = (number_opinion - 2) * p
    p_cA_list = np.arange(p + 1e-6, 0.1, 0.001)
    for p_cA in p_cA_list:
        for i in range(num_iter):
            p_list = np.random.random(number_opinion - 2) * p_cA 
            p_list = p_list/sum(p_list) * p_cAtilde
            while max(p_list) >= p_cA:
                diff = p_list - p
                p_list = p + diff *0.8
            print(p_cA)
            committed_fraction = np.hstack([p_cA, 0, p_list])
            committed_fraction_list.append(committed_fraction)

    parallel_attractors_approximation_S3(number_opinion, committed_fraction_list, des_file)
    return None

def attractors_approximation_big_small_committed(number_opinion, committed_fraction, des_file):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    length = 4 * number_opinion -3
    initial_state = np.zeros((length))

    initial_state[:2] = 0
    initial_state[2] = 1-sum(committed_fraction)
    initial_state[3:5] = committed_fraction 

    t = np.arange(0, 500, 0.01)
    coefficient = change_rule_approximation_S1(number_opinion)
    result = ode_stable_approximation(number_opinion, initial_state, length, coefficient)[-1, :3]
    data = np.hstack((committed_fraction, result))
    df_data = pd.DataFrame(data.reshape(1, len(data)))
    df_data.to_csv(des_file, index=None, header=None, mode='a')
    return None

def parallel_attractors_approximation_big_small_committed(number_opinion, p_cA_list, p_cAtilde):
    """TODO: Docstring for parallel_attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    des = '../data/big_small_committed/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'num_opinion={number_opinion}_p_cAtilde={p_cAtilde}.csv'
    committed_fraction_list = []
    for p_cA in p_cA_list:
        committed_fraction = np.array([p_cA, p_cAtilde])
        committed_fraction_list.append(committed_fraction)

    p = mp.Pool(cpu_number)
    p.starmap_async(attractors_approximation_big_small_committed, [(number_opinion, committed_fraction, des_file) for committed_fraction in committed_fraction_list]).get()
    p.close()
    p.join()
    return None

def compare_approximation_firstorder(number_opinion, committed_fraction):
    """TODO: Docstring for compare_approximation_firstorder.

    :arg1: TODO
    :returns: TODO

    """
    t = np.arange(0, 500, 0.01)
    single_fraction = np.zeros((number_opinion))
    single_fraction[1] = 1- sum(committed_fraction)
    length = 2**number_opinion -1 + number_opinion
    coefficient = change_rule(number_opinion)
    c_matrix = np.round(coefficient.reshape(length, length, length).transpose(2, 0, 1) , 15)
    mixed_fraction = np.zeros(( length-2*number_opinion))
    initial_state = np.hstack(([single_fraction, committed_fraction, mixed_fraction]))
    result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))

    "approximation S1"
    length = 4 * number_opinion - 3
    initial_state = np.zeros((length))
    initial_state[1] = 1- sum(committed_fraction)
    initial_state[3] = committed_fraction[0]
    initial_state[4] = sum(committed_fraction[2:])
    coefficient = change_rule_approximation_S1(number_opinion)
    result_S1 = odeint(mf_ode, initial_state, t, args=(length, coefficient))


    "approximation S3"
    length = 4 * number_opinion - 3
    small_committed = committed_fraction[2:]
    initial_state = np.zeros((length))
    initial_state[1] = 1- sum(committed_fraction)
    initial_state[3] = committed_fraction[0]
    initial_state[4] = sum(small_committed)
    coefficient = change_rule_approximation_S3(number_opinion, small_committed)
    result_S3 = odeint(mf_ode, initial_state, t, args=(length, coefficient))

    plt.plot(t, result[:, 0], linewidth=lw, alpha=alpha, label='$S_0$')
    plt.plot(t, result_S1[:, 0], linewidth=lw, alpha=alpha, label='$S_1$')
    plt.plot(t, result_S3[:, 0], linewidth=lw, alpha=alpha, label='$S_3$')
    plt.subplots_adjust(left=0.20, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.ylabel('$x_A$', fontsize=fontsize)
    plt.xlabel('$t$', fontsize=fontsize)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.show()


    return None

    


number_opinion = 6
digit = 10
committed_fraction = np.round(np.array([0.075, 0, 0.07, 0.06, 0.05, 0.04]), digit)
#compare_approximation_firstorder(number_opinion, committed_fraction)

p_list = np.round(np.arange(0.01, 0.1, 0.01), 3)
number_opinion_list = [4, 6, 7, 8, 9, 10]
num_iter = 10

for number_opinion in number_opinion_list:
    for p in p_list:
        #approximation_oneuncommitted(number_opinion, p, num_iter)
        pass

p_cAtilde_list = np.round(np.arange(0.2, 0.4, 0.01), 2)
number_opinion_list = [4, 5, 6, 7, 8]
#number_opinion = 8
for number_opinion in number_opinion_list:
    for p_cAtilde in p_cAtilde_list:
        p0 = round(p_cAtilde / (number_opinion - 2), 4)
        p_cA_list = np.round(np.arange(0, 0.4, 0.001), 3)
        parallel_attractors_approximation_big_small_committed(number_opinion, p_cA_list, p_cAtilde)
