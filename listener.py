import os
os.environ['OPENBLAS_NUM_THREADS'] ='1'

import sys
sys.path.insert(1, '/home/mac/RPI/research/')

import numpy as np 
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import time 
import pandas as pd
import multiprocessing as mp
from itertools import combinations, permutations
from cycler import cycler
import matplotlib as mpl
from scipy.signal import argrelextrema
import math



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

def change_rule(number_opinion):
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
    c_matrix = np.round(change_matrix.reshape(length, length, length).transpose(2, 0, 1) , 15)
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
    end = 100
    difference = 1
    single_fraction1 = single_fraction
    length = 2**number_opinion -1 + number_opinion
    mixed_fraction = np.zeros((length-2*number_opinion))
    while (difference) >=1e-3:
        t = np.arange(start, end, 0.01)
        initial_state = np.hstack(([single_fraction1, committed_fraction, mixed_fraction]))
        result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))
        single_fraction2 = result[-1, :number_opinion]
        difference = sum(abs(result[-1, :number_opinion] - result[-1000, :number_opinion]))
        mixed_fraction = result[-1, 2*number_opinion:]
        single_fraction1 = single_fraction2
    return single_fraction2

def attractors(N, committed_fraction, uncommitted_fraction, c_matrix, des_file):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    attractor = ode_stable(N, committed_fraction, uncommitted_fraction, c_matrix)
    data = np.hstack((committed_fraction, attractor))
    df_data = pd.DataFrame(data.reshape(1, len(data)))
    df_data.to_csv(des_file, index=None, header=None, mode='a')
    return None

def parallel_ode(N, committed_list, uncommitted_list, des_file):
    """TODO: Docstring for parallel_attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    length = 2**N -1 + N
    c_matrix = change_rule(N)
    p = mp.Pool(cpu_number)
    p.starmap_async(attractors, [(N, committed_fraction, uncommitted_fraction, c_matrix, des_file) for committed_fraction, uncommitted_fraction in zip(committed_list, uncommitted_list)]).get()
    p.close()
    p.join()
    return None


def permutation_different_time(a):
    """TODO: Docstring for sum_permutation_matrix.
    :a: TODO
    :returns: TODO
    """
    pairwise_S = lambda x: sum(np.prod(x, 0))
    height, width = np.shape(a)
    if height == 1:
        T = np.sum(a)
    else:
        T = permutation_different_time(a[:-1])  * sum(a[-1]) + (-1) ** (height-1) *  pairwise_S(a) * math.factorial(height-1)
        height_list = np.arange(height-1)
        for i in range(1, height-1):
            all_combination = list(combinations(height_list, height-1-i))
            for comb_i in all_combination:
                index_S = list(set(height_list) - set(comb_i))
                T += permutation_different_time(a[list(comb_i)])  * pairwise_S(np.vstack((a[index_S], a[-1]))) * (-1) ** i * math.factorial(i)
    return T

def mix_different_time(Q, number_opinion, x):
    """TODO: Docstring for function.

    :arg1: TODO
    :returns: TODO

    """
    mix_vector = np.zeros((number_opinion, number_opinion-1))
    mix_vector[:, 0] = x[-1] * (1-Q[-1]) + (sum(x[-1]) -x[-1]) * Q[-1]
    for i in range(number_opinion):
        mask = np.ones(np.size(Q, 1), dtype = bool)
        mask[i] = False
        Q_select = Q[:, mask]
        for mix_number in range(3, number_opinion+1):
            k = mix_number - 2
            Q_t = Q_select[1-mix_number:]
            Q_prod = permutation_different_time(Q_t)
            mix_vector[i, k] = x[-mix_number+1, i] * Q_prod 
            Q_select2 = Q[1-mix_number:, mask]
            for miss in range(mix_number-1):
                mask2 = np.ones(mix_number-1, dtype=bool)
                mask2[-miss-1] = False
                Q_t2 = Q_select2[mask2]

                Qx = np.vstack((Q_t2, x[-mix_number+1, mask]))
                Q_prod2 = permutation_different_time(Qx)
                mix_vector[i, k] +=  Q_prod2 * Q[-miss-1, i]

    return mix_vector.transpose()

def recursive_different_time(number_opinion, p, x, t, des_file):
    """TODO: Docstring for recursive_same_time.

    :number_opinion: TODO
    :x: TODO
    :p: TODO
    :returns: TODO

    """
    Q = p + x
    Q_history = np.zeros((number_opinion-1, number_opinion))
    x_history = np.zeros((number_opinion-1, number_opinion))
    Q_history[-1] = Q.copy()
    x_history[-1] = x.copy()
    x_evolution = np.zeros((t, number_opinion))
    Q_evolution = np.zeros((t, number_opinion))
    mix_evolution = np.zeros((t, number_opinion-1, number_opinion))
    mix_all = np.zeros((number_opinion-1, number_opinion))
    mix_sum = np.zeros((number_opinion)) 
    state_sum = x.sum() + p.sum()
    for i in range(0, t, 1):
        #print(i, x[0], Q.sum(), state_sum)
        x_evolution[i] = x
        Q_evolution[i] = Q
        mix_evolution[i] = mix_all
        x = Q * mix_sum + Q * x
        mix_all = mix_different_time(Q_history, number_opinion, x_history)
        uncommitted = np.hstack((x, np.sum(mix_all, 1) * np.array([1/mix_number for mix_number in range(2, number_opinion + 1)]) ))
        Q = x + p + np.sum(np.array([1/mix_number for mix_number in range(2, number_opinion + 1)]) * mix_all.transpose(), 1)
        Q = Q/sum(Q)
        uncommitted_norm = (1-sum(p))/sum(uncommitted)
        mix_all = mix_all * uncommitted_norm
        x = x * uncommitted_norm
        state_sum = uncommitted.sum() 
        mix_sum = np.sum(mix_all, 0)
        
        Q_history[:-1] = Q_history[1:].copy()
        Q_history[-1] = Q.copy()

        x_history[:-1] = x_history[1:].copy()
        x_history[-1] = x.copy()
    data = np.hstack((p, x_evolution[-1]))
    df_data = pd.DataFrame(data.reshape(1, len(data)))
    #df_data.to_csv(des_file, index=None, header=None, mode='a')
    return x_evolution, mix_evolution

def recursive_different_time_repeat(number_opinion, p, x_history, Q_history, mix_sum, t):
    """TODO: Docstring for recursive_same_time.

    :number_opinion: TODO
    :x: TODO
    :p: TODO
    :returns: TODO

    """
    x_evolution = np.zeros((t, number_opinion))
    Q_evolution = np.zeros((t, number_opinion))
    x = x_history[-1]
    Q = Q_history[-1]
    for i in range(0, t, 1):
        x_evolution[i] = x
        Q_evolution[i] = Q
        x = Q * mix_sum + Q * x
        mix_all = mix_different_time(Q_history, number_opinion, x_history)
        uncommitted = np.hstack((x, np.sum(mix_all, 1) * np.array([1/mix_number for mix_number in range(2, number_opinion + 1)]) ))
        Q = x + p + np.sum(np.array([1/mix_number for mix_number in range(2, number_opinion + 1)]) * mix_all.transpose(), 1)
        Q = Q/sum(Q)
        uncommitted_norm = (1-sum(p))/sum(uncommitted)
        x = x * uncommitted_norm
        mix_all = mix_all * uncommitted_norm
        mix_sum = np.sum(mix_all, 0)
        Q_history[:-1] = Q_history[1:].copy()
        Q_history[-1] = Q.copy()

        x_history[:-1] = x_history[1:].copy()
        x_history[-1] = x.copy()
    return x_evolution, Q_evolution, mix_sum

def recursive_stable(number_opinion, p, x, des_file):
    """TODO: Docstring for ode_stable.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    t = 100
    difference = 1
    Q = p + x
    Q_history = np.zeros((number_opinion-1, number_opinion))
    x_history = np.zeros((number_opinion-1, number_opinion))
    Q_history[-1] = Q.copy()
    x_history[-1] = x.copy()
    mix_sum = np.zeros((number_opinion)) 
    while (difference) >= 1e-3:
        x_evolution, Q_evolution, mix_sum = recursive_different_time_repeat(number_opinion, p, x_history, Q_history, mix_sum, t)
        x_history = x_evolution[-(number_opinion-1):]
        Q_history = Q_evolution[-(number_opinion-1):]
        difference = sum(abs(x_evolution[-1] - x_evolution[0]))
    data = np.hstack((p, x_evolution[-1]))
    df_data = pd.DataFrame(data.reshape(1, len(data)))
    df_data.to_csv(des_file, index=None, header=None, mode='a')
    return x_evolution

def parallel_recursive(number_opinion, committed_list, uncommitted_list, des_file):
    """TODO: Docstring for parallel_attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(recursive_stable, [(number_opinion, committed_fraction, uncommitted_fraction, des_file) for committed_fraction, uncommitted_fraction in zip(committed_list, uncommitted_list)]).get()
    p.close()
    p.join()
    return None


def pA_critical_N(pA_list, N_list, pAtilde, ode):
    """TODO: Docstring for pA_critical_N.
    :returns: TODO

    """
    if ode:
        des = f'../data/listener_only/ode/pA_critical_N/'
    else:
        des = f'../data/listener_only/recursive/pA_critical_N/'
    if not os.path.exists(des):
        os.makedirs(des)
    for N in N_list:
        des_file = des + f'pAtilde={pAtilde}_N={N}.csv'
        p_list = []
        x_list = []
        p0 = pAtilde / (N -2) 
        for pA in pA_list:
            p = np.hstack((pA, 0, np.ones(N-2) * p0))
            x = np.hstack((0, 1-sum(p), np.zeros(N-2)))
            p_list.append(p)
            x_list.append(x)
        if ode:
            parallel_ode(N, p_list, x_list, des_file)
        else:
            parallel_recursive(N, p_list, x_list, des_file)
    return None

def pA_critical_N_fluctuation(pA_list, N_list, pAtilde, max_fluctuation, seed_list, ode):
    """TODO: Docstring for pA_critical_N.
    :returns: TODO

    """
    if ode:
        des = f'../data/listener_only/ode/pA_critical_N_fluctuation/'
    else:
        des = f'../data/listener_only/recursive/pA_critical_N_fluctuation/'
    if not os.path.exists(des):
        os.makedirs(des)
    for N in N_list:
        p0 = pAtilde / (N -2) 
        for seed in seed_list:
            des_file = des + f'pAtilde={pAtilde}_N={N}_maxfluc={max_fluctuation}_seed={seed}.csv'
            p_list = []
            x_list = []
            for pA in pA_list:
                p_fluctuate = np.random.RandomState(seed).uniform(0, max_fluctuation, N-2)
                pAtilde_fluctuate = pAtilde + p_fluctuate
                pAtilde_fluctuate = pAtilde_fluctuate / sum(pAtilde_fluctuate) * pAtilde
                p = np.hstack((pA, 0, pAtilde_fluctuate))
                x = np.hstack((0, 1-sum(p), np.zeros(N-2)))
                p_list.append(p)
                x_list.append(x)
            if ode:
                parallel_ode(N, p_list, x_list, des_file)
            else:
                parallel_recursive(N, p_list, x_list, des_file)
    return None

def pA_critical_N_fluctuation_dirichlet(pA_list, N_list, pAtilde, alpha_list, seed_list, ode):
    """TODO: Docstring for pA_critical_N.
    :returns: TODO

    """
    if ode:
        des = f'../data/listener_only/ode/pA_critical_N_fluctuation_dirichlet/'
    else:
        des = f'../data/listener_only/recursive/pA_critical_N_fluctuation_dirichlet/'
    if not os.path.exists(des):
        os.makedirs(des)
    for N in N_list:
        p0 = pAtilde / (N -2) 
        for seed in seed_list:
            for alpha in alpha_list:
                p_list = []
                x_list = []
                des_file = des + f'pAtilde={pAtilde}_N={N}_alpha={alpha}_seed={seed}.csv'
                for pA in pA_list:
                    p_fluctuate = np.random.RandomState(seed).dirichlet(np.ones(N-2) * alpha)
                    pAtilde_fluctuate = p_fluctuate * pAtilde
                    p = np.hstack((pA, 0, pAtilde_fluctuate))
                    x = np.hstack((0, 1-sum(p), np.zeros(N-2)))
                    p_list.append(p)
                    x_list.append(x)
                if ode:
                    parallel_ode(N, p_list, x_list, des_file)
                else:
                    parallel_recursive(N, p_list, x_list, des_file)
    return None


def attractors_Nchange(N, committed_fraction, uncommitted_fraction, des_file):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    length = 2**N -1 + N
    c_matrix = change_rule(N)
    attractor = ode_stable(N, committed_fraction, uncommitted_fraction, c_matrix)
    data = np.hstack((committed_fraction[:4], attractor[:4]))
    df_data = pd.DataFrame(data.reshape(1, len(data)))
    df_data.to_csv(des_file, index=None, header=None, mode='a')
    return None

def parallel_ode_Nchange(N_list, committed_list, uncommitted_list, des_file):
    """TODO: Docstring for parallel_attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :returns: TODO

    """
    p = mp.Pool(cpu_number)
    p.starmap_async(attractors_Nchange, [(N, committed_fraction, uncommitted_fraction, des_file) for N, committed_fraction, uncommitted_fraction in zip(N_list, committed_list, uncommitted_list)]).get()
    p.close()
    p.join()
    return None

def pA_critical_N_lowerbound(N_list, pAtilde, ode):
    """TODO: Docstring for pA_critical_N.
    :returns: TODO

    """
    if ode:
        des = f'../data/listener_only/ode/pA_critical_N_lowerbound/'
    else:
        des = f'../data/listener_only/recursive/pA_critical_N_lowerbound/'
    if not os.path.exists(des):
        os.makedirs(des)
    for N in N_list:
        des_file = des + f'pAtilde={pAtilde}_N={N}.csv'
        p0 = pAtilde / (N -2) 
        pmax_list = np.arange(p0, pAtilde+0.0001, 0.001)
        number_opinion_list = []
        p_list = []
        x_list = []
        for pmax in pmax_list:
            n_max = int(np.round(pAtilde/pmax, 13))  # round-off error
            pA = pmax + 0.001
            pC = pAtilde - pmax * n_max
            p = np.hstack((pA, 0, pC, np.ones(n_max) * pmax))
            number_opinion = n_max + 3
            x = np.hstack((0, 1-sum(p), 0, np.zeros(n_max)))
            p_list.append(p)
            x_list.append(x)
            number_opinion_list.append(number_opinion)
        if ode:
            parallel_ode_Nchange(number_opinion_list, p_list, x_list, des_file)
        else:
            parallel_recursive_Nchange(number_opinion_list, p_list, x_list, des_file)
    return None


def evolution_ode(number_opinion, p, x):
    """TODO: Docstring for evolution_ode.

    :number_opinion: TODO
    :p: TODO
    :x: TODO
    :returns: TODO

    """
    c_matrix = change_rule(number_opinion)
    length = 2**number_opinion -1 + number_opinion
    initial = np.zeros((length))
    initial[:number_opinion] =  x
    initial[number_opinion: 2*number_opinion] = p

    dt = 0.01
    t = 300
    tstep = np.arange(0, t, dt)
    result = odeint(mf_ode, initial, tstep, args=(length, c_matrix))

    t3 = time.time()
    plt.plot(tstep, result[:, 0], linestyle='-.', linewidth=2, alpha = 0.8, label='$A$')
    plt.plot(tstep, result[:, 1], linestyle='-.', linewidth=2, alpha = 0.8, label='$B$')
    plt.plot(tstep, result[:, 2], linestyle='-.', linewidth=2, alpha = 0.8, label='$C$')
    plt.plot(tstep, result[:, 3], linestyle='-.', linewidth=2, alpha = 0.8, label='$D$')
    plt.plot(tstep, result[:, 4], linestyle='-.', linewidth=2, alpha = 0.8, label='$E$')
    plt.plot(tstep, result[:, 5], linestyle='-.', linewidth=2, alpha = 0.8, label='$F$')
    plt.xlabel('t', fontsize=fontsize)
    plt.ylabel('$x$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.82, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)

    plt.legend(frameon=False, fontsize = legendsize)
    plt.locator_params(nbins=5)
    plt.show()





number_opinion = 5
p = np.array([0.058, 0, 0.1/3, 0.1/3, 0.1/3])
x = np.array([0.0, 1-p.sum(), 0, 0, 0.0])
t = 100
t1 = time.time()
#a, b = recursive_different_time(number_opinion, p, x, t, '../data/data.csv')
t2 = time.time()

pA_list = np.arange(0.001, 0.12, 0.001)
N_list = np.arange(3, 8, 1)
pAtilde = 0.02
ode = 1
#pA_critical_N(pA_list, N_list, pAtilde, ode)

max_fluctuation = 0.01
seed_list = np.arange(100).tolist()
#pA_critical_N_fluctuation(pA_list, N_list, pAtilde, max_fluctuation, seed_list, ode)
pAtilde_list = [0.06, 0.14, 0.18]
for pAtilde in pAtilde_list:
    #pA_critical_N_lowerbound(N_list, pAtilde, ode)
    pass

alpha_list = np.array([1])
seed_list = [0]

#pA_critical_N_fluctuation_dirichlet(pA_list, N_list, pAtilde, alpha_list, seed_list, ode)



number_opinion = 6
p = np.array([0.055, 0, 0.05, 0.03, 0.01, 0.01])
p = np.array([0.055, 0, 0.025, 0.025, 0.025, 0.025])
p = np.array([0.051, 0, 0.05, 0.03, 0.01, 0.01])
x = np.array([0, 1-sum(p), 0, 0, 0, 0])
#evolution_ode(number_opinion, p, x)

"""
"mean-field"

number_opinion = 6
p = np.array([0.1, 0, 0.025, 0.025, 0.025, 0.025])
x = np.array([0, 1-sum(p), 0, 0, 0, 0])
dt = 0.01
t = 100

x_evolution, mix_evolution = recursive_different_time(number_opinion, p, x, t, '')

c_matrix = change_rule(number_opinion)
length = 2**number_opinion -1 + number_opinion
initial = np.zeros((length))
initial[:number_opinion] =  x
initial[number_opinion: 2*number_opinion] = p


tstep = np.arange(0, t, dt)
result2 = odeint(mf_ode, initial, tstep, args=(length, c_matrix))

t3 = time.time()
plt.plot(np.arange(t), x_evolution[:, 2], '#66c2a5', linestyle='-', linewidth=2, alpha = 0.8, label='Recursive')
#plt.plot(np.arange(t), result1[:, 0], 'tab:blue', linestyle='--', linewidth=2, alpha=0.8, label='$\Delta t=1$')
plt.plot(tstep, result2[:, 2], '#fc8d62', linestyle='-.', linewidth=2, alpha = 0.8, label='ODEs')
plt.xlabel('t', fontsize=fontsize)
plt.ylabel('$x_C$', fontsize=fontsize)
plt.xticks(fontsize=ticksize)
plt.yticks(fontsize=ticksize)
plt.subplots_adjust(left=0.18, right=0.82, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)

plt.legend(frameon=False, fontsize = legendsize)
plt.locator_params(nbins=5)
plt.show()
"""
