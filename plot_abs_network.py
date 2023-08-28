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
mpl.rcParams['axes.prop_cycle'] = (cycler(color=[i for i in ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'] for j in range(1)] ))




number_opinion = 3
fontsize = 22
ticksize= 15
legendsize = 15
alpha = 0.8
lw = 3

def p_fluctuation(N, number_opinion, p, fluctuate_seed, sigma_p, sigma_pu):
    """TODO: Docstring for p_fluctuation.

    :network_type: TODO
    :N: TODO
    :seed: TODO
    :d: TODO
    :fluctuation_seed: TODO
    :sigma_p: TODO
    :sigma_pu: TODO
    :returns: TODO

    """
    minority = number_opinion - 1
    seed_p = fluctuate_seed
    seed_pu = fluctuate_seed
    while True:
        p_fluctuate = np.random.RandomState(seed_p).normal(p, sigma_p*p, minority)
        p_fluctuate = p_fluctuate / p_fluctuate.sum() * p*minority
        nc_minority =  np.array(np.round(N * p_fluctuate), int)
        if nc_minority.min() > 0 and nc_minority.sum() == round(N * p * minority):
            break
        else:
            seed_p += 1

    pu = (1-pA - p * minority) / minority
    while True:
        pu_fluctuate = np.random.RandomState(seed_pu).normal(pu, sigma_pu*pu, minority)
        pu_fluctuate = pu_fluctuate / pu_fluctuate.sum() * pu*minority
        if pu_fluctuate.min() > 0 and pu_fluctuate.max() < pu * minority:
            break
        else:
            seed_pu += 1
    return p_fluctuate.std(), p_fluctuate.max(), pu_fluctuate.std(), pu_fluctuate.max()

    



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
    #plt.show()
    plt.savefig(save_file)
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

def cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p, fluctuation_seed=None, sigma_p=None, sigma_pu=None):
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
            xA_mean, ratio = cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p)
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
    plt.savefig(save_file)
    plt.close('all')
    #plt.show()

    return None

def plot_ratio_pA(network_type, N_list, net_seed, d, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list, fluctuation_seed, sigma_p, sigma_pu):
    """TODO: Docstring for plot_x_pA.

    :arg1: TODO
    :returns: TODO

    """
    for number_opinion, p in zip(number_opinion_list, p_list):
        R_pA = []
        for pA in pA_list:
            xA_mean, ratio = cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p, fluctuation_seed, sigma_p, sigma_pu)
            R_pA.append(ratio)
        plt.plot(pA_list,  R_pA, '-', alpha=alpha, linewidth=lw, label=f'$m={number_opinion}$')
    save_des = f'../report/report041322/' + network_type + f'_N={N}_d={d}_netseed={net_seed}_p={p_list[0]}_R_pA.png'
    plt.rc('font', size=ticksize)
    plt.xlabel('$P_A$', fontsize=fontsize)
    plt.ylabel('$R_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params(nbins=6)
    #plt.savefig(save_des)
    #plt.close('all')
    #plt.show()

    return None

def plot_ratio_pA_fluctuate(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA_list, p, fluctuation_seed, sigma_p_list, sigma_pu_list):
    """TODO: Docstring for plot_x_pA.

    :arg1: TODO
    :returns: TODO

    """
    for sigma_p, sigma_pu in zip(sigma_p_list, sigma_pu_list):
        pf_std,  pf_max, puf_std, puf_max = p_fluctuation(N, number_opinion, p, fluctuation_seed, sigma_p, sigma_pu)
        R_pA = []
        for pA in pA_list:
            xA_mean, ratio = cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p, fluctuation_seed, sigma_p, sigma_pu)
            R_pA.append(ratio)
        plt.plot(pA_list,  R_pA, '-', alpha=alpha, linewidth=lw, label='$\\sigma_{p}=$' + f'{round(pf_std, 3)}' + ', $p_{m}=$' + f'{round(pf_max, 3)}')
    save_des = f'../report/report041322/' + network_type + f'_N={N}_d={d}_netseed={net_seed}_m={number_opinion}_p={p_list[0]}_R_pA_fluctuation.png'
    plt.rc('font', size=ticksize)
    plt.xlabel('$P_A$', fontsize=fontsize)
    plt.ylabel('$R_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize * 0.8, markerscale=1.0, loc=2)
    plt.locator_params(nbins=6)
    plt.savefig(save_des)
    plt.close('all')
    #plt.show()

    return None

def plot_x_pA_average_diffsize(network_type, N_list, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA_list, p):
    """TODO: Docstring for plot_x_pA.

    :arg1: TODO
    :returns: TODO

    """
    for N in N_list:
        interaction_number = N * 1000
        xA_pA = []
        for pA in pA_list:
            xA_mean, ratio = cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p)
            xA_pA.append(xA_mean)
        plt.plot(pA_list,  xA_pA, '-', alpha=alpha, linewidth=lw, label=f'$N={N}$')

    p_cAtilde = p * (number_opinion - 1)
    des_mf = '../data/big_small_committed/'
    des_file_mf = des_mf + f'num_opinion={number_opinion+1}_p_cAtilde={p_cAtilde}.csv'
    data_mf = np.array(pd.read_csv(des_file_mf, header=None))
    pA_mf = data_mf[:, 0]
    index_sort_pA = np.argsort(pA_mf)
    pA_sort = pA_mf[index_sort_pA]
    p_Atilde = data_mf[index_sort_pA, 1]
    xA_mf = data_mf[index_sort_pA, 2]
    index_plot = np.where(pA_sort <= pA_list[-1])[0]
    plt.plot(pA_sort[index_plot], xA_mf[index_plot] + pA_sort[index_plot], alpha=alpha, linewidth=lw, label='mf-ode')

    save_des = f'../report/report022622/' + network_type + f'_d={d}_netseed={net_seed}_m={number_opinion}_p={p}_nA_pA.png'
    plt.rc('font', size=ticksize)
    plt.ylabel('$n_A^{(s)}$', fontsize=fontsize)
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

def plot_ratio_pA_diffsize(network_type, N_list, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA_list, p):
    """TODO: Docstring for plot_x_pA.

    :arg1: TODO
    :returns: TODO

    """
    for N in N_list:
        interaction_number = N * 1000
        R_pA = []
        for pA in pA_list:
            xA_mean, ratio = cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p)
            R_pA.append(ratio)
        plt.plot(pA_list,  R_pA, '-', alpha=alpha, linewidth=lw, label=f'$N={N}$')
    save_des = f'../report/report022622/' + network_type + f'_d={d}_netseed={net_seed}_m={number_opinion}_p={p}_R_pA.png'
    plt.rc('font', size=ticksize)
    plt.xlabel('$P_A$', fontsize=fontsize)
    plt.ylabel('$R_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params(nbins=6)
    plt.savefig(save_des)
    plt.close('all')
    #plt.show()

    return None

def plot_x_pA_average_networks(network_type_list, N, net_seed_list, d_list, interaction_number, number_opinion, comm_seed_list, pA_list, p):
    """TODO: Docstring for plot_x_pA.

    :arg1: TODO
    :returns: TODO

    """
    for network_type, net_seed, d in zip(network_type_list, net_seed_list, d_list):
        if network_type == 'complete':
            labels = 'complete'
        elif network_type == 'ER':
            labels = f'$\\langle k \\rangle  = {int(2 * d / N)}$'

        interaction_number = N * 1000
        xA_pA = []
        for pA in pA_list:
            xA_mean, ratio = cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p)
            xA_pA.append(xA_mean)

        plt.plot(pA_list,  xA_pA, '-', alpha=alpha, linewidth=lw, label=labels)

    p_cAtilde = p * (number_opinion - 1)
    des_mf = '../data/big_small_committed/'
    des_file_mf = des_mf + f'num_opinion={number_opinion+1}_p_cAtilde={p_cAtilde}.csv'
    data_mf = np.array(pd.read_csv(des_file_mf, header=None))
    pA_mf = data_mf[:, 0]
    index_sort_pA = np.argsort(pA_mf)
    pA_sort = pA_mf[index_sort_pA]
    p_Atilde = data_mf[index_sort_pA, 1]
    xA_mf = data_mf[index_sort_pA, 2]
    index_plot = np.where(pA_sort <= pA_list[-1])[0]
    plt.plot(pA_sort[index_plot], xA_mf[index_plot] + pA_sort[index_plot], alpha=alpha, linewidth=lw, label='mf-ode')

    save_des = f'../report/report022622/' 
    save_des = f'../manuscript/031422/figure/' 
    save_file = save_des + f'N={N}_m={number_opinion}_p={p}_nA_pA.png'
    plt.rc('font', size=ticksize)
    plt.ylabel('$\\langle n_A^{(s)} \\rangle $', fontsize=fontsize)
    plt.xlabel('$P_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params(nbins=6)
    plt.savefig(save_file)
    plt.close('all')
    #plt.show()

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
                xA_mean, ratio = cal_xA_pA_ave_ratio(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p)
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








N = 100000
N = 10000
N = 1000
interaction_number = N * 1000


network_type = 'SF'
net_seed = [98, 98]
d = [2.5, 0, 3]
net_seed = [79, 79]
d = [3.5, 0, 4]
net_seed = [5, 5]
d = [2.1, 0, 2]


net_seed = 0
d = 16000

network_type = 'complete'
net_seed = 0
d = 0

network_type = 'ER'
net_seed = 0
d = 8000





number_opinion = 5
pA = 0.07
pA_list = [0.02, 0.03, 0.04, 0.05]
p = 0.01
plot_opinion_list = ['A', 'B', 'C', 'D', 'E']
plot_opinion_list = ['A', 'B', 'C']
plot_opinion_list = ['A', 'B']
comm_seed_list = [1]
plot_opinion_list = ['A']
comm_seed_list = np.arange(50)  
for pA in pA_list:
    #plot_x_t(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA, p, plot_opinion_list)
    pass
pA_list = np.round(np.arange(0.01, 0.1, 0.01), 2)
network_type = 'ER'
d_list = [2000, 3000, 4000, 8000, 16000]
net_seed_list = [0, 1, 0, 0, 0]
pA_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]

network_type = 'SF'
d_list = [[2.1, 0, 2], [2.5, 0, 3], [3.5, 0, 4]]
net_seed_list = [[5, 5], [98, 98], [79, 79]]


network_type = 'complete'
d_list = [0]
net_seed_list = [0]

network_type = 'ER'
d_list = [3000, 4000, 8000, 16000]
net_seed_list = [1, 0, 0, 0]


pA_list = np.round(np.arange(0.01, 0.16, 0.01), 2)
number_opinion_list = [2, 3, 4, 7]
p_list = [0.06, 0.03, 0.02, 0.01] 


for d, net_seed in zip(d_list, net_seed_list):
    #plot_x_pA(network_type, N, net_seed, d, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list)
    #plot_x_pA_average(network_type, N, net_seed, d, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list)
    #plot_ratio_pA(network_type, N, net_seed, d, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list)
    pass


network_type = 'complete'
N_list = [1000, 10000, 100000]
d = 0
net_seed = 0
number_opinion = 7
p = 0.01
#plot_ratio_pA_diffsize(network_type, N_list, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA_list, p)
#plot_x_pA_average_diffsize(network_type, N_list, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA_list, p)
N = 10000
network_type_list = ['ER', 'ER', 'ER', 'ER', 'complete']
net_seed_list = [1, 0, 0, 0, 0]
d_list = [3000, 4000, 8000, 16000, 0]
d_list = [30000, 40000, 80000, 160000, 0]
#plot_x_pA_average_networks(network_type_list, N, net_seed_list, d_list, interaction_number, number_opinion, comm_seed_list, pA_list, p)
number_opinion_list = [2, 3, 4, 7]
p_list = [0.06, 0.03, 0.02, 0.01]
comm_seed_list = np.arange(50)
plot_ratio_pAc_m(network_type_list, N, net_seed_list, d_list, interaction_number, number_opinion_list, comm_seed_list, pA_list, p_list)


N = 1000

network_type = 'complete' 
net_seed = 0
d = 0

network_type = 'ER'
d = 8000
net_seed = 0

interaction_number = N * 1000
number_opinion = 5
p = 0.015
comm_seed_list = np.arange(50)
pA_list = np.round(np.arange(0.01, 0.11, 0.01), 2) 
fluctuation_seed = 0
sigma_p = 0
sigma_pu = 0
sigma_p_list = [0, 0.1, 1, 10, 100]
sigma_p_list = [0, 0.1, 100, 1, 10]
sigma_pu_list = [0, 0, 0, 0, 0]
#plot_ratio_pA_fluctuate(network_type, N, net_seed, d, interaction_number, number_opinion, comm_seed_list, pA_list, p, fluctuation_seed, sigma_p_list, sigma_pu_list)
