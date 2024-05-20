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
import networkx as nx



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

def network_generate(network_type, N, beta, betaeffect, seed, d=None):
    """TODO: Docstring for network_generate.

    :arg1: TODO
    :returns: TODO

    """
    if network_type == 'complete':
        G = nx.complete_graph(N)
    if network_type == '1D':
        G = nx.grid_graph(dim=[N], periodic=True)
    if network_type == '2D':
        G = nx.grid_graph(dim=[int(np.sqrt(N)),int(np.sqrt(N))], periodic=True)
    if network_type == '3D':
        G = nx.grid_graph(dim=[round(N**(1/3)), round(N**(1/3)), round(N**(1/3))], periodic=True)
    if network_type == '2D_disorder' or network_type == '3D_disorder':
        if network_type == '2D_disorder':
            G = nx.grid_graph(dim=[int(np.sqrt(N)),int(np.sqrt(N))], periodic=True)
        if network_type == '3D_disorder':
            G = nx.grid_graph(dim=[round(N**(1/3)), round(N**(1/3)), round(N**(1/3))], periodic=True)
        A = nx.to_numpy_array(G)
        #t = time.time()
        #modifer = int((t-int(t)) * 1e9)
        modifer = 1000
        A = A * np.random.RandomState(seed + modifer).uniform(0, 1,  (N, N))  
        A = np.triu(A, 0) + np.triu(A, 1).transpose()
        A = np.array(A > (1-d), dtype=int)  # 1-d is the edge remove rate
        G = nx.from_numpy_matrix(A)
    elif network_type == 'RR':
        G = nx.random_regular_graph(d, N, seed)
    elif network_type == 'ER':
        #G = nx.fast_gnp_random_graph(N, d, seed)
        m = d
        G = nx.gnm_random_graph(N, m, seed)
    elif network_type == 'BA':
        m = d
        G = nx.barabasi_albert_graph(N, m, seed)
    elif network_type == 'SF':
        
        gamma, kmax, kmin = d
        G = generate_SF(N, seed, gamma, kmax, kmin)
        '''
        kmax = int(kmin * N ** (1/(gamma-1))) 
        probability = lambda k: (gamma - 1) * kmin**(gamma-1) * k**(-gamma)
        k_list = np.arange(kmin, 10 *kmax, 0.001)
        p_list = probability(k_list)
        p_list = p_list/np.sum(p_list)
        degree_seq = np.array(np.round(np.random.RandomState(seed=seed[0]).choice(k_list, size=N, p=p_list)), int)
        kmin, gamma, kmax = d
        degree_seq = np.array(((np.random.RandomState(seed[0]).pareto(gamma-1, N) + 1) * kmin), int)
        degree_max = np.max(degree_seq)
        if degree_max > kmax:
            degree_seq[degree_seq>kmax] = kmax
        else:
            degree_seq[degree_seq == degree_max] = kmax
        i = 0
        while np.sum(degree_seq)%2:
            i+=1
            #degree_seq[-1] = int(np.round(np.random.RandomState(seed=i).choice(k_list, size=1, p=p_list))) 
            degree_seq[-1] = int((np.random.RandomState(seed=N+i).pareto(gamma-1, 1) + 1) * kmin)
            degree_max = np.max(degree_seq)
            if degree_max > kmax:
                degree_seq[degree_seq>kmax] = kmax
            else:
                degree_seq[degree_seq == degree_max] = kmax

        G = nx.configuration_model(degree_seq, seed=seed[1])
        G = nx.Graph(G)  # remove parallel edges
        G.remove_edges_from(list(nx.selfloop_edges(G)))  # remove self loops (networkx version is not the newest one)
        '''
    elif network_type == 'star':
        G = nx.star_graph(N-1)
    elif network_type == 'RGG':
        G = nx.generators.geometric.random_geometric_graph(N, d, seed=seed)

    elif network_type == 'real':
        A, M, N = load_data(seed)
        #A = A_from_data(seed%2, M)
        A = np.heaviside(A, 0) # unweighted network
        G = nx.from_numpy_matrix(A)
    elif network_type == 'SBM_ER':
        N_group = N
        p = d
        G = nx.stochastic_block_model(N_group, p, seed=seed)

    elif network_type == 'degree_seq':
        G = nx.configuration_model(d, seed=seed)
 

    if nx.is_connected(G) == False:
        print('more than one component')
        G = G.subgraph(max(nx.connected_components(G), key=len))
    #A = np.array(nx.adjacency_matrix(G).todense()) 
    A = nx.to_numpy_array(G)
    if betaeffect:
        beta_eff, _ = betaspace(A, [0])
        weight = beta/ beta_eff
    else:
        weight = beta
    A = A * weight
    A_index = np.where(A>0)
    A_interaction = A[A_index]
    index_i = A_index[0] 
    index_j = A_index[1] 
    degree = np.sum(A>0, 1)
    cum_index = np.hstack((0, np.cumsum(degree)))
    return A, A_interaction, index_i, index_j, cum_index

def generate_SF(N, seed, gamma, kmax, kmin):
    """generate scale-free network using configuration model with given gamma, kmin, kmax. 

    :N: TODO
    :seed: TODO
    :gamma: TODO
    :kmin: TODO
    :kmax: TODO
    :returns: TODO

    """
    p = lambda k: k ** (float(-gamma))
    k = np.arange(kmin, N, 1)
    pk = p(k) / np.sum(p(k))
    random_state = np.random.RandomState(seed[0])
    if kmax == N-1 or kmax == N-2:
        degree_seq = random_state.choice(k, size=N, p=pk)
    elif kmax == 0 or kmax == 1:
        degree_try = random_state.choice(k, size=1000000, p=pk)
        k_upper = int(np.sqrt(N * np.mean(degree_try)))
        k = np.arange(kmin, k_upper+1, 1)
        pk = p(k) /np.sum(p(k))
        degree_seq = random_state.choice(k, size=N, p=pk)

    i = 0
    while np.sum(degree_seq)%2:
        i+=1
        degree_seq[-1] = np.random.RandomState(seed=seed[0]+N+i).choice(k, size=1, p=pk)

    degree_original = degree_seq.copy()

    G = nx.empty_graph(N)
    "generate scale free network using configuration model"
    no_add = 0
    degree_change = 1
    j = 0
    while np.sum(degree_seq) and no_add < 10:

        stublist = nx.generators.degree_seq._to_stublist(degree_seq)
        M = len(stublist)//2  # the number of edges

        random_state = np.random.RandomState(seed[1] + j)
        random_state.shuffle(stublist)
        out_stublist, in_stublist = stublist[:M], stublist[M:]
        if degree_change == 0:
            no_add += 1
        else:
            no_add = 0
        G.add_edges_from(zip(out_stublist, in_stublist))

        G = nx.Graph(G)  # remove parallel edges
        G.remove_edges_from(list(nx.selfloop_edges(G)))  # remove self loops (networkx version is not the newest one)
        if nx.is_connected(G) == False:
            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        degree_alive = np.array([G.degree[i] if i in G.nodes() else 0 for i in range(N)])
        degree_former = np.sum(degree_seq)
        degree_seq = degree_original - degree_alive
        degree_now = np.sum(degree_seq)
        degree_change = degree_now-degree_former
        j += 1
        if kmax == 1 or kmax == N-2:
            break
    return G

