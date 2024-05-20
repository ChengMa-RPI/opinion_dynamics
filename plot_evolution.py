import sys
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
from helper_function import network_generate

mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:red', 'tab:olive', 'tab:cyan']) 
mpl.rcParams['axes.prop_cycle'] = (cycler(color=[i for i in ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'] for j in range(1)] ))

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
    if network_type == 'complete':
        N_actual = N
    else:
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
    save_des = '../manuscript/031422/figure/'
    if len(plot_opinion_list) == 1:
        save_file = save_des + network_type + f'_N={N}_d={d}_netseed={net_seed}_number_opinion={number_opinion}_pA={pA}_p={p}_x' + plot_opinion[0] + '.png'
        plt.ylabel(f'$n_{plot_opinion_list[0]}$', fontsize=fontsize)
    else:
        save_file = save_des + network_type + f'_N={N}_d={d}_netseed={net_seed}_number_opinion={number_opinion}_pA={pA}_p={p}_x_commseed={comm_seed_list[0]}.png'
        plt.ylabel('$n$', fontsize=fontsize)
        plt.legend(frameon=False, fontsize = legendsize)

    plt.xlabel('$T$', fontsize=fontsize)
    plt.rc('font', size=ticksize)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.locator_params(nbins=6)
    #plt.savefig(save_file)
    plt.show()
    plt.close('all')
    return None

def helper_cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p, fluctuation_seed=None, sigma_p=None, sigma_pu=None):
    """TODO: Docstring for plot_x_pA.

    :arg1: TODO
    :returns: TODO

    """
    if sigma_p is None and sigma_pu is None:
        des = f'../data/' + network_type + f'/N={N}_d={d}_netseed={net_seed}/actual_simulation/number_opinion={number_opinion}/interaction_number={interaction_number}_pA={pA}_p={p}/'
    else:
        des = f'../data/' + network_type + f'/N={N}_d={d}_netseed={net_seed}/actual_simulation_fluctuate/fluctuate_seed={fluctuation_seed}_sigma_p={sigma_p}_sigma_pu={sigma_pu}/number_opinion={number_opinion}/interaction_number={interaction_number}_pA={pA}_p={p}/'

    Ni_list = []
    for comm_seed in comm_seed_list:
        des_file = des + f'comm_seed={comm_seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        t = data[:, 0]
        Ni = data[:, 1:]
        Ni_list.append(Ni)
    Ni_list = np.stack(Ni_list, axis=0)
    xi_list = Ni_list/N
    xA_list = xi_list[:, :, 0] + pA
    xA_mean = np.mean(xA_list[:, -1])
    ratio = np.sum(xA_list[:, -1] > 0.5) / len(comm_seed_list)
    return xA_mean, ratio

def plot_x_pA_average(network_type, N, net_seed, d, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list):
    """TODO: Docstring for plot_x_pA.

    :arg1: TODO
    :returns: TODO

    """
    for number_opinion, p in zip(number_opinion_list, p_list):
        xA_pA = []
        for pA in pA_list:
            xA_mean, ratio = helper_cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p)
            xA_pA.append(xA_mean)
        plt.plot(pA_list,  xA_pA, '-', alpha=alpha, linewidth=lw, label=f'$m={number_opinion}$')

    save_des = f'../manuscript/031422/figure/' 
    save_file = save_des + network_type + f'_N={N}_d={d}_netseed={net_seed}_p={p_list[0]}_nA_pA.png'
    plt.rc('font', size=ticksize)
    plt.xlabel('$P_A$', fontsize=fontsize)
    plt.ylabel('$\\langle n_A^{(s)} \\rangle$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params(nbins=6)
    #plt.savefig(save_file)
    plt.show()
    plt.close('all')

    return None

def plot_ratio_pA(network_type, N_list, net_seed, d, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list, fluctuation_seed=None, sigma_p=None, sigma_pu=None):
    """TODO: Docstring for plot_x_pA.

    :arg1: TODO
    :returns: TODO

    """
    for number_opinion, p in zip(number_opinion_list, p_list):
        R_pA = []
        for pA in pA_list:
            xA_mean, ratio = helper_cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p, fluctuation_seed, sigma_p, sigma_pu)
            R_pA.append(ratio)
        plt.plot(pA_list,  R_pA, '-', alpha=alpha, linewidth=lw, label=f'$m={number_opinion}$')
    plt.rc('font', size=ticksize)
    plt.xlabel('$P_A$', fontsize=fontsize)
    plt.ylabel('$R_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params(nbins=6)
    save_des = f'../report/report041322/' + network_type + f'_N={N}_d={d}_netseed={net_seed}_p={p_list[0]}_R_pA.png'
    #plt.savefig(save_des)
    plt.show()
    plt.close('all')

    return None

def plot_ratio_pAc_m(network_type_list, N, net_seed_list, d_list, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list):
    """TODO: Docstring for plot_x_pA.

    :arg1: TODO
    :returns: TODO

    """
    interaction_number = N * 1000
    linestyles = ['-', '--', 'dashdot'] * 2
    markerstyles = ['o', '*', 's', 'd', 'p']
    markersizes = [8, 10, 7, 8, 8]
    for (i, network_type), net_seed, d in zip(enumerate(network_type_list), net_seed_list, d_list):
        if network_type == 'complete':
            labels = 'complete'
        elif network_type == 'ER':
            labels = f'$\\langle k \\rangle  = {int(2 * d / N)}$'
        PA_critical = []
        for number_opinion, p in zip(number_opinion_list, p_list):
            R_pA = []
            for pA in pA_list:
                xA_mean, ratio = helper_cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p)
                R_pA.append(ratio)
            R_pA = np.array(R_pA)
            PA_critical.append(pA_list[np.where(R_pA >1/2)[0][0]])
        plt.plot(number_opinion_list,  PA_critical,  linestyle=linestyles[i], alpha=alpha, linewidth=lw, marker=markerstyles[i], markersize=markersizes[i], label=labels)
    save_des = '../manuscript/031422/figure/'
    save_file = save_des + f'N={N}_p={p_list[0]}_pA_critical_m_ratio.png'
    plt.rc('font', size=ticksize)
    plt.xlabel('$m$', fontsize=fontsize)
    plt.ylabel('$P_A^{(c)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params(nbins=6)
    #plt.savefig(save_file)
    #plt.close('all')
    plt.show()

    return None

def heatmap_pAc_m(network_type, N, net_seed_list, d_list, k_ave, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list):
    """TODO: Docstring for plot_x_pA.

    :arg1: TODO
    :returns: TODO

    """
    interaction_number = N * 1000
    data = []
    for net_seed, d in zip(net_seed_list, d_list):
        PA_critical = []
        for number_opinion, p in zip(number_opinion_list, p_list):
            R_pA = []
            for pA in pA_list:
                xA_mean, ratio = helper_cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p)
                R_pA.append(ratio)
            R_pA = np.array(R_pA)
            PA_critical.append(pA_list[np.where(R_pA >1/2)[0][0]])
        data.append(PA_critical)
    data = data[::-1]
    ax = sns.heatmap(data, linewidth=0, cmap='YlGnBu', cbar_kws={'label': '$P_A^{(c)}$'})
    ax.figure.axes[-1].yaxis.label.set_size(22)


    ax.set_yticks(np.arange(len(k_ave)) + 0.5)
    ax.set_yticklabels(k_ave[::-1], rotation=-0)
    ax.set_xticks(np.arange(len(number_opinion_list)) + 0.5)
    ax.set_xticklabels(number_opinion_list, rotation=0)


    save_des = '../figure0213/'
    PA_tilde = round(number_opinion_list[0] * p_list[0], 2)
    save_file = save_des + f'heatmap_{network_type}_N={N}_pAtilde={PA_tilde}_pA_critical_m_ratio.png'
    plt.rc('font', size=ticksize)
    plt.xlabel('$m$', fontsize=fontsize)
    plt.ylabel('$\\langle k \\rangle $', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.15, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    #plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params(nbins=6)
    plt.savefig(save_file)
    plt.close('all')
    #plt.show()

    return None


########################################################
""" Parameters Explained
network_type, N, d, net_seed are used in the function network_generate to generate networks.
network_type: complete--complete graph, ER--ER network, SF--scale free network
N: the number of nodes/agents in the network to exchange opinions
d: the parameter to control the network generation, 
0 for complete graph (no meaning), 
an integer for ER network, determining the edge density, k=d * 2 / N, 
a list for Scale-free network, [gamma, kmax, kmin], gamma is the scale free exponent parameter, kmax=0 means no restriction on the maximum of degree
net_seed: random seed to generate graph

interaction_number: the number of interactions between agents to exchange opinions, usually a very large number to make sure that the system reaches the equilibrium
number_opinion: the number of single opinions, "m"
comm_seed: the random seed to select agents for exchanging opinions
pA: the fraction of committed agents supporting opinion A
p: the fraction of committed agents supporting minority B, C, D, ...
"""
########################################################
"Figure 10"
network_type = 'ER'
N = 1000
interaction_number = 1000 * 1000
number_opinion = 5
comm_seed_list = np.arange(50)  
pA_list = [0.02, 0.03, 0.04]
p = 0.01
plot_opinion_list = ['A']  # plot the evolution of opinion A


"Figure 10a-c"
d = 3000  
net_seed = 1
"Figure 10d-f"
d = 4000
net_seed = 0
"Figure 10g-i"
d = 8000
net_seed = 0

for pA in pA_list:
    #plot_x_t(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p, plot_opinion_list)
    pass
########################################################


########################################################
"Figure 11"
network_type = 'ER'
N = 1000
interaction_number = 1000 * 1000
net_seed_list = [1, 0, 0]
d_list = [3000, 4000, 8000]

number_opinion_list = [2, 3, 4, 5,  7]

pA_list = np.round(np.arange(0.01, 0.16, 0.01), 2)
p_list = [0.06, 0.03, 0.02, 0.01] 

for d, net_seed in zip(d_list, net_seed_list):
    "Figure 11 a - c"
    #plot_x_pA_average(network_type, N, net_seed, d, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list)
    "Figure 11 d - f"
    #plot_ratio_pA(network_type, N, net_seed, d, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list)
    pass
########################################################


#######################################################
"Figure 12"
network_type_list = ['ER', 'ER', 'ER', 'ER', 'complete']
net_seed_list = [1, 0, 0, 0, 0]  
interaction_number = 1000 * 1000  
number_opinion_list = [2, 3, 4, 7]  
p_list = [0.06, 0.03, 0.02, 0.01]  # p_0, the committed minority, such that P_tilde{A} is fixed as 0.06

comm_seed_list = np.arange(50)
pA_list = np.round(np.arange(0.01, 0.16, 0.01), 2)  # pA to identify the critical point.


"Figure 12b, larger networks with 10000 nodes"
N = 10000
d_list = [30000, 40000, 80000, 160000, 0]
"Figure 12a, smaller networks with 1000 nodes"
N = 1000 
d_list = [3000, 4000, 8000, 16000, 0]  # parameter to control density of ER network, 0 if network is complete/full connected


#plot_ratio_pAc_m(network_type_list, N, net_seed_list, d_list, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list)
########################################################



#######################################################
"Figure12 heatmap"
comm_seed_list = np.arange(50)
pA_list = np.round(np.arange(0.01, 0.16, 0.01), 2)  # pA to identify the critical point.
N = 1000 
network_type = 'ER'
d_list = [3000, 4000, 8000, 16000]  
k_ave = [6, 8, 16, 32]
net_seed_list = [1, 0, 0, 0]  
interaction_number = N * 1000  
number_opinion_list = [2, 3, 4, 7]  
p_list = [0.06, 0.03, 0.02, 0.01]  # p_0, the committed minority, such that P_tilde{A} is fixed as 0.06
number_opinion_list = [2, 3, 4, 7]  
p_list = [0.06, 0.03, 0.02, 0.01]  # p_0, the committed minority, such that P_tilde{A} is fixed as 0.06
heatmap_pAc_m(network_type, N, net_seed_list, d_list, k_ave, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list)
