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

from function_simulation_ode_mft import transition_rule, all_state, actual_simulation, actual_simulation_save_all, change_rule, mf_ode


cpu_number = 5
fontsize = 22
ticksize= 15
legendsize = 16
alpha = 0.8
lw = 3

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']) )


def parallel_approximation_BKS(number_opinion, N, interaction_number, data_point, seed_list, pA, p, ng_type):
    """TODO: Docstring for parallel_actual_simlation.

    :arg1: TODO
    :returns: TODO

    """
    na = int(N * pA)
    nb = int(N * p)
    nA = int(N - na - nb * (number_opinion - 1))
    state_list = all_state(number_opinion)
    state_single = state_list[0: number_opinion]
    state_committed = state_list[number_opinion: 2*number_opinion]
    state_Atilde = state_committed[1:] 
    initial_state = ['a'] * na + ['A'] * nA  + functools.reduce(operator.iconcat, [[i] * nb for i in state_Atilde], [])

    if ng_type == 'original':
        des = f'../data/approximation_BKS/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_p={p}/'
    elif ng_type == 'LO':
        des = f'../data/approximation_BKS/actual_simulation_LO/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_p={p}/'

    if not os.path.exists(des):
        os.makedirs(des)
    p = mp.Pool(cpu_number)
    p.starmap_async(actual_simulation_save_all, [(N, interaction_number, data_point, initial_state, state_list, seed, des, ng_type) for seed in seed_list]).get()
    p.close()
    p.join()
    return None

def mft_evolution_BKS(number_opinion, pA, p, ng_type):
    """TODO: Docstring for attractors.

    :number_opinion: TODO
    :committed_fraction: TODO
    :single_fraction: TODO
    :returns: TODO

    """
    if ng_type == 'original':
        des = f'../data/approximation_BKS/mfe/number_opinion={number_opinion}/'
    elif ng_type == 'LO':
        des = f'../data/approximation_BKS/mfe_LO/number_opinion={number_opinion}/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'pA={pA}_p={p}.csv'
    length = 2**number_opinion -1 + number_opinion
    c_matrix = change_rule(number_opinion, ng_type)
    committed_fraction = np.hstack(([pA, p * np.ones(number_opinion-1)]))
    single_fraction = np.hstack(([1-sum(committed_fraction), np.zeros(number_opinion-1)]))
    mixed_fraction = np.zeros((length-2*number_opinion))
    initial_state = np.hstack(([single_fraction, committed_fraction, mixed_fraction]))
    t = np.arange(0, 100, 0.01)
    result = odeint(mf_ode, initial_state, t, args=(length, c_matrix))[::10]
    data = np.hstack((t[::10].reshape(len(t[::10]), 1), result))
    df_data = pd.DataFrame(data)
    df_data.to_csv(des_file, index=None, header=None, mode='a')
    return None

def recursive_BKS_LO(number_opinion, pA, p, T):
    """TODO: Docstring for recursive_BKS.

    :number_opinion: TODO
    :pA: TODO
    :p: TODO
    :returns: TODO

    """
    des = f'../data/approximation_BKS/recursive_LO/number_opinion={number_opinion}/'
    if not os.path.exists(des):
        os.makedirs(des)
    des_file = des + f'pA={pA}_p={p}.csv'

    m = number_opinion - 1
    x = np.zeros((T+1, 2*m+1))
    x[0, 0] = 1-p*m - pA

    Q0 = np.zeros(T)
    Q1 = np.zeros(T)
    for t in range(0, T, 1):
        Q0[t] = pA + np.sum([1/(s+1) * x[t, s] for s in range(0, m+1, 1)])
        Q1[t] = p + np.sum([s/(s+1)/m * x[t, s] for s in range(1, m+1, 1)]) + np.sum([1/m * x[t, s] for s in range(m+1, 2*m+1, 1)])
        Q0[t] = Q0[t] / (Q0[t] + m * Q1[t])
        Q1[t] = Q1[t] / (Q0[t] + m * Q1[t])
        x[t+1, 0] = np.sum(x[t, :m+1]) * Q0[t]
        x[t+1, m+1] = np.sum(x[t, m+1:2*m+1] * np.arange(1, m+1, 1)) * Q1[t] + np.sum([s * x[t, s]  for s in range(1, m+1, 1)]) * Q1[t]
        for s in range(1, m+1):
            x[t+1, s] = x[t, s-1] * Q1[t] * (m-s+1) + x[t, m+s] * Q0[t]
        for s in range(m+2, 2*m+1, 1):
            x[t+1, s] = x[t, s-1] * Q1[t] * (2*m-s+1) 
        x[t+1] =  x[t+1] / np.sum(x[t+1]) * x[0].sum()
    data = np.hstack((np.arange(0, T+1, 1).reshape(T+1, 1), x))
    df_data = pd.DataFrame(data)
    df_data.to_csv(des_file, index=None, header=None, mode='a')

    return x





ng_type = 'LO'
number_opinion = 4
N_list = np.array([1000, 5000, 10000, 50000, 100000])
N_list = np.array([1000, 10000, 100000])
interaction_number_list = N_list* 100
data_point = 1000
seed_list = np.arange(100)
pA = 0.1
p_list = [0.01]
for N, interaction_number in zip(N_list, interaction_number_list):
    for p in p_list:
        #parallel_approximation_BKS(number_opinion, N, interaction_number, data_point, seed_list, pA, p, ng_type)
        pass

number_opinion = 4
pA = 0.1
p = 0.01
mft_evolution_BKS(number_opinion, pA, p, ng_type)
T = 100
#x = recursive_BKS_LO(number_opinion, pA, p, T)
