import sys
sys.path.insert(1, '/home/mac/RPI/research/')

import numpy as np 
import os
import matplotlib.pyplot as plt
import networkx as nx
import time
from numpy import linalg as LA
import pandas as pd 
import scipy.io
import seaborn as sns
from cycler import cycler
import matplotlib as mpl
import itertools
from mpl_toolkits.mplot3d import Axes3D
from mutual_framework import network_generate

mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:red', 'tab:olive', 'tab:cyan']) 
mpl.rcParams['axes.prop_cycle'] = (cycler(color=[i for i in ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'] for j in range(2)] ))




number_opinion = 3
fontsize = 22
ticksize= 15
legendsize = 15
alpha = 0.8
lw = 3

def plot_x_t(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p, plot_opinion_list):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, net_seed, d)
    N_actual = len(A)
    des = f'../data/' + network_type + f'/N={N}_d={d}_netseed={net_seed}/actual_simulation/number_opinion={number_opinion}/interaction_number={interaction_number}_pA={pA}_p={p}/'
    Ni_list = []
    for comm_seed in comm_seed_list:
        des_file = des + f'comm_seed={comm_seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        t = data[:, 0]
        Ni = data[:, 1:]
        Ni_list.append(Ni)
    Ni_list = np.stack(Ni_list, axis=0)
    xi_list = Ni_list/N_actual
    for plot_opinion in plot_opinion_list:
        plot_index = ord(plot_opinion) - 65
        plt.plot(t/N_actual, xi_list[:, :, plot_index].transpose(), color=colors[plot_index], alpha=alpha, label=plot_opinion, linewidth=lw)
    if len(plot_opinion_list) == 1:
        save_des = f'../report/report022622/' + network_type + f'_N={N}_d={d}_netseed={net_seed}_number_opinion={number_opinion}_pA={pA}_p={p}_x' + plot_opinion[0] + '.png'
        plt.ylabel(f'$n_{plot_opinion_list[0]}$', fontsize=fontsize)
    else:
        save_des = f'../report/report022622/' + network_type + f'_N={N}_d={d}_netseed={net_seed}_number_opinion={number_opinion}_pA={pA}_p={p}_x_commseed={comm_seed_list[0]}.png'
        plt.ylabel('$n$', fontsize=fontsize)
        plt.legend(frameon=False, fontsize = legendsize)

    plt.xlabel('$T$', fontsize=fontsize)
    plt.rc('font', size=ticksize)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.locator_params(nbins=6)
    #plt.show()
    plt.savefig(save_des)
    plt.close('all')
    return None

def detect_x_pA(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA_list, p):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    A, A_interaction, index_i, index_j, cum_index = network_generate(network_type, N, 1, 0, net_seed, d)
    N_actual = len(A)
    xA_pA_1 = []
    xA_pA_2 = []
    for pA in pA_list:
        des = f'../data/' + network_type + f'/N={N}_d={d}_netseed={net_seed}/actual_simulation/number_opinion={number_opinion}/interaction_number={interaction_number}_pA={pA}_p={p}/'
        Ni_list = []
        for comm_seed in comm_seed_list:
            des_file = des + f'comm_seed={comm_seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            t = data[:, 0]
            Ni = data[:, 1:]
            Ni_list.append(Ni)
        Ni_list = np.stack(Ni_list, axis=0)
        xi_list = Ni_list/N_actual
        xA_list = xi_list[:, :, 0] + pA
        index_dominate = np.where(xA_list[:, -1] > 0.5)[0]
        index_nondominate = np.where(xA_list[:, -1] < 0.5)[0]
        if 0 < len(index_dominate) < len(comm_seed_list):
            xA_pA_1.append(np.mean(xA_list[index_dominate, -1]))
            xA_pA_2.append(np.mean(xA_list[index_nondominate, -1]))
        
        else:
            xA_pA_1.append(np.mean(xA_list[:, -1]))
            xA_pA_2.append(np.mean(xA_list[:, -1]))
    return xA_pA_1, xA_pA_2

def plot_x_pA(network_type, N, net_seed, d, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list):
    """TODO: Docstring for plot_x_pA.

    :arg1: TODO
    :returns: TODO

    """
    for number_opinion, p in zip(number_opinion_list, p_list):
        xA_pA_1, xA_pA_2 = detect_x_pA(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA_list, p)
        plt.plot(pA_list,  xA_pA_1, '-', alpha=alpha, linewidth=lw, label=f'$m={number_opinion}$')
        plt.plot(pA_list,  xA_pA_2, '--', alpha=alpha, linewidth=lw)
        plt.ylabel('$n_A^{(s)}$', fontsize=fontsize)
    save_des = f'../report/report022622/' + network_type + f'_N={N}_d={d}_netseed={net_seed}_p={p_list[0]}_nA_pA.png'
    plt.rc('font', size=ticksize)
    plt.xlabel('$P_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params(nbins=6)
    plt.savefig(save_des)
    plt.close('all')
    #plt.show()

    return None



N = 1000




network_type = 'SF'
net_seed = [98, 98]
d = [2.5, 0, 3]
net_seed = [79, 79]
d = [3.5, 0, 4]
net_seed = [5, 5]
d = [2.1, 0, 2]


network_type = 'ER'
net_seed = 1
d = 3000

net_seed = 0
d = 16000

network_type = 'complete'
net_seed = 0
d = 0




interaction_number = 1000000
number_opinion = 5
pA = 0.07
pA_list = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
p = 0.01
plot_opinion_list = ['A', 'B', 'C', 'D', 'E']
comm_seed_list = [0]
plot_opinion_list = ['A']
comm_seed_list = np.arange(10)
for pA in pA_list:
    plot_x_t(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p, plot_opinion_list)
    pass
pA_list = np.round(np.arange(0.01, 0.1, 0.01), 2)
network_type = 'ER'
d_list = [2000, 3000, 4000, 8000, 16000]
net_seed_list = [0, 1, 0, 0, 0]
pA_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

network_type = 'SF'
d_list = [[2.1, 0, 2], [2.5, 0, 3], [3.5, 0, 4]]
net_seed_list = [[5, 5], [98, 98], [79, 79]]

network_type = 'ER'
d_list = [3000, 4000, 8000, 16000]
net_seed_list = [1, 0, 0, 0]

pA_list = np.round(np.arange(0.01, 0.16, 0.01), 2)
number_opinion_list = [2, 3, 4, 7]
p_list = [0.06, 0.03, 0.02, 0.01] 


for d, net_seed in zip(d_list, net_seed_list):
    #plot_x_pA(network_type, N, net_seed, d, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list)
    pass
