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

mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:red', 'tab:olive', 'tab:cyan']) 
mpl.rcParams['axes.prop_cycle'] = (cycler(color=[i for i in ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'] for j in range(1)] ))




number_opinion = 3
fontsize = 22
ticksize= 15
legendsize = 15
alpha = 0.8
lw = 3

def plot_actual_simulation(number_opinion, N, interaction_number, interval, seed_list, pA, p, plot_opinion):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    des = f'../data/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_p={p}/'
    NA_list = []
    NB_list = []
    NC_list = []

    for seed in seed_list:
        des_file = des + f'seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        t = data[:, 0]
        NA = data[:, 1]
        NB = data[:, 2]
        #NC = data[:, 3]
        NC = 0
        NA_list.append(NA)
        NB_list.append(NB)
        NC_list.append(NC)
    NA_list = np.vstack((NA_list)).transpose()
    NB_list = np.vstack((NB_list)).transpose()
    NC_list = np.vstack((NC_list)).transpose()
    xA_list = NA_list/N
    xB_list = NB_list/N
    xC_list = NC_list/N
    if plot_opinion == 'A':
        plt.plot(t/N, xA_list[:, :], color='tab:green', alpha=alpha)
        #plt.plot(t/N, np.mean(xA_list, 1), color='k', alpha=alpha, linewidth=lw)
        #plt.plot(t/N, np.mean(xA_list, 1), alpha=alpha, linewidth=lw)
        plt.ylabel('$x_A^{(s)}$', fontsize=fontsize)
        save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xA.png'
    elif plot_opinion == 'B':
        plt.plot(t, xB_list, color='tab:green', alpha=alpha)
        plt.plot(t, np.mean(xB_list, 1), color='k', alpha=alpha, linewidth=lw)
        plt.ylabel('$x_B^{(s)}$', fontsize=fontsize)
        save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xB.png'

    elif plot_opinion == 'C':
        plt.plot(t, xC_list, color='tab:orange', alpha=alpha)
        plt.plot(t, np.mean(xC_list, 1), color='k', alpha=alpha, linewidth=lw)
        plt.ylabel('$x_C^{(s)}$', fontsize=fontsize)
        save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xC.png'
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.rc('font', size=ticksize)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.xlabel('$T$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, 0.))
    plt.locator_params(nbins=6)
    #plt.savefig(save_des)
    #plt.close('all')

    return None

def compare_actual_mfe(number_opinion, N, interaction_number, interval, seed_list, pA, p, plot_opinion):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    des = f'../data/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_p={p}/'
    NA_list = []
    NB_list = []
    NC_list = []

    for seed in seed_list:
        des_file = des + f'seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        t = data[:, 0]
        NA = data[:, 1]
        NB = data[:, 2]
        NC = data[:, 3]
        NA_list.append(NA)
        NB_list.append(NB)
        NC_list.append(NC)
    NA_list = np.vstack((NA_list)).transpose()
    NB_list = np.vstack((NB_list)).transpose()
    NC_list = np.vstack((NC_list)).transpose()
    xA_list = NA_list/N
    xB_list = NB_list/N
    xC_list = NC_list/N
    des_mfe = f'../data/approximation_compare/number_opinion={number_opinion}/pA={pA}_p={p}.csv'
    data_mfe = np.array(pd.read_csv(des_mfe, header=None))
    xA_mfe = data_mfe[:, 0]
    xB_mfe = data_mfe[:, 1]
    xC_mfe = data_mfe[:, 2]
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    if plot_opinion == 'A':
        line1 = ax1.plot(t, np.mean(xA_list, 1), color='tab:red', linewidth=lw, alpha=alpha, label='Actual interaction')
        line2 = ax2.plot(xA_mfe[:100], '--', color = 'tab:blue', linewidth=lw, alpha=alpha, label='MFE')
        ax1.set_ylabel('$x_A^{(s)}$', fontsize=fontsize)
        save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xA_compare.png'
    elif plot_opinion == 'B':
        line1 = ax1.plot(t, np.mean(xB_list, 1), color='tab:green', linewidth=lw, alpha=alpha, label='Actual interaction')
        line2 = ax2.plot(xB_mfe[:100], '--', color = 'tab:blue', linewidth=lw, alpha=alpha, label='MFE')
        ax1.set_ylabel('$x_B^{(s)}$', fontsize=fontsize)
        save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xB_compare.png'

    elif plot_opinion == 'C':
        line1 = ax1.plot(t, np.mean(xC_list, 1), color='tab:orange', linewidth=lw, alpha=alpha, label='Actual interaction')
        line2 = ax2.plot(xC_mfe[:100]/(number_opinion-2), '--', color = 'tab:blue', linewidth=lw, alpha=alpha, label='MFE')
        ax1.set_ylabel('$x_C^{(s)}$', fontsize=fontsize)
        ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xC_compare.png'

    plt.rc('font', size=ticksize)
    ax1.set_xlabel('T', fontsize=fontsize)
    ax1.tick_params(axis='x', labelsize=ticksize, colors='tab:red') 
    ax1.tick_params(axis='y', labelsize=ticksize) 
    ax2.tick_params(axis='x',labelsize=ticksize, colors='tab:blue') 
    ax1.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.subplots_adjust(left=0.18, right=0.90, wspace=0.25, hspace=0.25, bottom=0.15, top=0.90)
    lns = line1+line2
    labs = [l.get_label() for l in lns]
    plt.legend(lns, labs, frameon=False, fontsize = legendsize)
    plt.locator_params(nbins=6)
    plt.savefig(save_des)
    plt.close('all')
    return None

def consensus_time_size(number_opinion, N_list, interaction_number_list, interval, seed_list, pA, p):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    T_list = []
    for N, interaction_number in zip(N_list, interaction_number_list):
        des = f'../data/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_p={p}/'
        NA_list = []
        NB_list = []
        NC_list = []

        for seed in seed_list:
            des_file = des + f'seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            t = data[:, 0]
            NA = data[:, 1]
            NB = data[:, 2]
            NC = data[:, 3]
            NA_list.append(NA)
            NB_list.append(NB)
            NC_list.append(NC)
        NA_list = np.vstack((NA_list)).transpose()
        NB_list = np.vstack((NB_list)).transpose()
        NC_list = np.vstack((NC_list)).transpose()
        xA_list = NA_list/N
        xB_list = NB_list/N
        xC_list = NC_list/N
        xA_mean = np.mean(xA_list, 1)
        xB_mean = np.mean(xB_list, 1)
        xC_mean = np.mean(xC_list, 1)
        T = t[np.where(xB_mean <1e-3)[0][0]]
        T = t[np.where(xA_mean >= 1/2 * xA_mean[-1])[0][0]]
        T_list.append(T)
    T_list = np.array(T_list)
    m, b = np.polyfit(N_list, T_list, 1)
    data_fit = np.array([number_opinion, pA, p, m, b])
    df_data = pd.DataFrame(data_fit.reshape(1, len(data_fit)))
    des_file = '../data/actual_simulation/consensustime_fit.csv'
    df_data.to_csv(des_file, index=None, header=None, mode='a')
 
    plt.plot(N_list, T_list, 'o')
    plt.plot(N_list, m * N_list + b, '--', linewidth=2, alpha=alpha, label=f'$p_0={p}$')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('system size $N$', fontsize=fontsize)
    plt.ylabel('consensus time $T$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.locator_params(nbins=6)
    return m, b

def consensus_time_slope(M_list):
    """TODO: Docstring for consensus_time_slope.
    :returns: TODO

    """
    des_file = '../data/actual_simulation/consensustime2_fit.csv'
    des_file = '../data/actual_simulation/consensustime_fit.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    number_opinion = data[:, 0]
    pA = data[:, 1]
    p = data[:, 2]
    uncommitted_size = 1 - pA - p * (number_opinion - 2)
    m = data[:, 3]
    b = data[:, 4]
    #number_opinion_list = np.unique(number_opinion)
    number_opinion_list = M_list
    pA_list = np.unique(pA)
    for pA_plot in pA_list:
        for M in number_opinion_list:
            index = np.where((number_opinion == M) & (pA == pA_plot))[0]
            if len(index):
                #plt.plot(uncommitted_size[index], m[index], 'o', label=f'$M={int(M)}$' + f'_$P_A={pA_plot}$')
                uncommitted_sort = np.argsort(uncommitted_size[index])
                plt.plot(uncommitted_size[index][uncommitted_sort], m[index][uncommitted_sort], 'o', label= f'$P_A={pA_plot}$')
    plt.legend(frameon=False, fontsize = legendsize)
    plt.xlabel('uncommitted size $P_u$', fontsize=fontsize)
    plt.ylabel('slope $k$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.locator_params(nbins=6)
    return None

def Tc_N_onecommitted(number_opinion, N_list, interaction_number_list, seed_list, pA):
    """TODO: Docstring for consensus_time_slope.
    :returns: TODO

    """
    consensus_time_list = np.zeros((np.size(N_list), np.size(seed_list)))
    for N_i_comb, interaction_number in zip(enumerate(N_list), interaction_number_list):
        i, N = N_i_comb
        des = f'../data/actual_simulation/onecommitted/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}/'
        xA_s = 1 - pA
        NA_list = []
        NB_list = []
        for j, seed in enumerate(seed_list):
            des_file = des + f'seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            t = data[:, 0] 
            NA = data[:, 1]
            NB = data[:, 2]
            xA = NA/N
            xB = NB/N
            consensus_time = np.where(np.abs(xA - xA_s) < 1e-9)[0][0] / 10
            consensus_time_list[i, j] = consensus_time

    plt.semilogx(N_list, np.mean(consensus_time_list, 1), 'o')
    plt.legend(frameon=False, fontsize = legendsize)
    plt.xlabel('$N$', fontsize=fontsize)
    plt.ylabel('$T_c$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.legend(frameon=False, fontsize = legendsize)
    #plt.locator_params(nbins=6)
    return None

def compare_actual_mfe_onecommitted(number_opinion, N_list, interaction_number_list, data_point, seed_list, pA):
    """TODO: Docstring for consensus_time_slope.
    :returns: TODO

    """
    xA_actual = np.zeros((np.size(N_list), np.size(seed_list), data_point))
    xB_actual = np.zeros((np.size(N_list), np.size(seed_list), data_point))
    t_actual = np.zeros((np.size(N_list), np.size(seed_list), data_point))
    for N_i_comb, interaction_number in zip(enumerate(N_list), interaction_number_list):
        i, N = N_i_comb
        des_actual = f'../data/actual_simulation/onecommitted/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}/'
        xA_s = 1 - pA
        NA_list = []
        NB_list = []
        for j, seed in enumerate(seed_list):
            des_actual_file = des_actual + f'seed={seed}.csv'
            data = np.array(pd.read_csv(des_actual_file, header=None))
            t = data[:, 0] / N
            NA = data[:, 1]
            NB = data[:, 2]
            xA = NA/N
            xB = NB/N
            xA_actual[i, j] = xA
            xB_actual[i, j] = xB
            t_actual[i, j] = t 

    des_mfe = f'../data/mft_evolution/onecommitted/number_opinion={number_opinion}/'
    des_mfe_file = des_mfe + f'pA={pA}.csv'
    data = np.array(pd.read_csv(des_mfe_file, header=None))
    t_mfe = data[:, 0]
    xA_mfe = data[:, 1]
    xB_mfe = data[:, 2]

    plt.plot(t_actual[0].transpose(), xA_actual[0].transpose(), alpha=0.7, color='tab:blue')
    plt.plot(t_mfe, xA_mfe, alpha = 0.8, linestyle='--', color='tab:red', linewidth=lw)
    for i, N in enumerate(N_list):
        #plt.plot(t_actual[i, 0].transpose(), np.mean(xA_actual[i].transpose(), 1), alpha=0.7, label=f'N={N}')
        pass

    #plt.plot(t_actual[-1].transpose(), xA_actual[-1].transpose(), alpha=0.7, color='tab:green')
    x = 1/ (xA_s - xA_mfe)
    index = np.where(x > 0)[0]
    #plt.semilogx(x[index], t_mfe[index], alpha = 0.8, color='tab:red', linewidth=lw)
    plt.xlim(-1, 30)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.xlabel('$t$', fontsize=fontsize)
    plt.ylabel('$x_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.legend(frameon=False, fontsize = legendsize)
    #plt.locator_params(nbins=6)
    return None

def Tc_m_actual(number_opinion, N, interaction_number, seed_list, pA, p):
    """TODO: Docstring for consensus_time_slope.
    :returns: TODO

    """
    consensus_time_list = np.zeros((np.size(seed_list)))
    des = f'../data/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_p={p}/'
    NA_list = []
    NB_list = []
    NC_list = []
    for j, seed in enumerate(seed_list):
        des_file = des + f'seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        t = data[:, 0] / N
        NA = data[:, 1]
        NB = data[:, 2]
        NC = data[:, 3]
        xA = NA/N
        xB = NB/N
        xA_s = np.mean(xA[-20:])
        consensus_time = np.where(xA - xA_s > 0)[0][0] / 10
        consensus_time_list[j] = consensus_time

    #plt.semilogx(number_opinion, np.mean(consensus_time_list), 'o')
    plt.plot(t, xA)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.xlabel('$m$', fontsize=fontsize)
    plt.ylabel('$T_c$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.legend(frameon=False, fontsize = legendsize)
    #plt.locator_params(nbins=6)
    return None

def Tc_m_mft(number_opinion, pA, p):
    """TODO: Docstring for consensus_time_slope.
    :returns: TODO

    """
    des = f'../data/mft_evolution/number_opinion={number_opinion}/'
    NA_list = []
    NB_list = []
    NC_list = []
    des_file = des + f'pA={pA}_p={p}.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    t = data[:, 0] 
    xA = data[:, 1]
    xB = data[:, 2]
    xC = data[:, 3]
    xA_s = np.mean(xA[-20:])
    consensus_time = t[np.where(np.abs(xA - xA_s) < 1e-5)[0][0] ]

    plt.plot(number_opinion, consensus_time, 'o', markersize=8, color='tab:blue')
    #plt.plot(t, xA)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.xlabel('$m$', fontsize=fontsize)
    plt.ylabel('$T_c$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.locator_params('x', nbins=6)
    return None

def plot_actual_simulation_two_opinion(number_opinion, N, interaction_number, interval, seed_list, pA, pB, xA, plot_opinion):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    #des = f'../data/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_pB={pB}_switch_direction=' + switch_direction + '/'
    des = f'../data/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_pB={pB}_xA={xA}/'
    NA_list = []
    NB_list = []
    for seed in seed_list:
        des_file = des + f'seed={seed}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        t = data[:, 0]
        NA = data[:, 1]
        NB = data[:, 2]
        NA_list.append(NA)
        NB_list.append(NB)
    NA_list = np.vstack((NA_list)).transpose()
    NB_list = np.vstack((NB_list)).transpose()
    xA_list = NA_list/N
    xB_list = NB_list/N
    if plot_opinion == 'A':
        plt.plot(t/N, np.mean(xA_list[:, :], 1), color='tab:green', alpha=alpha)
        #plt.ylabel('$x_A^{(s)}$', fontsize=fontsize)
        plt.ylabel('$x_A$', fontsize=fontsize)
        save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xA.png'
    elif plot_opinion == 'B':
        plt.plot(t/N, xB_list, color='tab:blue', alpha=alpha)
        plt.plot(t/N, np.mean(xB_list, 1), color='tab:blue', alpha=alpha, linewidth=lw)
        #plt.ylabel('$x_B^{(s)}$', fontsize=fontsize)
        plt.ylabel('$x_B$', fontsize=fontsize)
        save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xB.png'

        #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.rc('font', size=ticksize)
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.xlabel('$t$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, 0.))
    plt.locator_params(nbins=6)
    #plt.savefig(save_des)
    #plt.close('all')
    return None

def switching_time_N(number_opinion, N_list, interaction_number, interval, seed_list, pA, pB, switch_direction, switch_threshold, approx_integer):
    """TODO: Docstring for switching_time.

    :arg1: TODO
    :returns: TODO

    """
    #des = f'../data/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_pB={pB}_switch_direction=' + switch_direction + '/'
    T_list = np.zeros((len(N_list)))
    for i_N, N in enumerate(N_list):
        des = f'../data/actual_simulation/number_opinion={number_opinion}/approx_integer=' + approx_integer + f'/N={N}_pA={pA}_pB={pB}_switch_direction=' + switch_direction + '/'
        T_switching = []
        for seed in seed_list:
            des_file = des + f'seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            t = data[:, 0]
            NA = data[:, 1]
            NB = data[:, 2]
            if switch_direction == 'A-B':
                switching_index = np.where(NB / N - switch_threshold > -1e-5)[0]
            elif switch_direction == 'B-A':
                switching_index = np.where(NA / N - switch_threshold > -1e-5)[0]

            if len(switching_index):
                T_switching.append(t[switching_index[0]] / N )
                #T_switching.append(t[switching_index[0]] )
        if len(T_switching) < len(seed_list):
            print('simulation time is not enough', len(T_switching))

        T_list[i_N] = np.mean(T_switching)
    #plt.semilogy(N_list, T_list, 'o-', label=f'$P_B={pB}$')
    if approx_integer == 'round':
        round_method = 'round half down'
    elif approx_integer == 'floor':
        round_method = 'round down'
    elif approx_integer == 'ceil':
        round_method = 'round up'
    #plt.semilogy(N_list, T_list, 'o-', label=round_method)
    plt.semilogx((N_list), T_list, 'o-', label=round_method)
    plt.rc('font', size=ticksize)
    plt.xlabel('$N$', fontsize=fontsize)
    plt.ylabel('$T_{\\mathrm{switching}}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    #plt.legend(frameon=False, fontsize = legendsize, markerscale=1)
    #plt.show()

    return T_switching

def switching_time_N_multi(number_opinion, N_list, seed_list, pA, p0, switch_direction, switch_threshold, approx_integer):
    """TODO: Docstring for switching_time.

    :arg1: TODO
    :returns: TODO

    """
    T_list = np.zeros((len(N_list)))
    for i_N, N in enumerate(N_list):
        des = f'../data/actual_simulation/number_opinion={number_opinion}/approx_integer=' + approx_integer + f'/N={N}_pA={pA}_p0={p0}_switch_direction=' + switch_direction + '/'
        T_switching = []
        for seed in seed_list:
            des_file = des + f'seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            t = data[:, 0]
            NA = data[:, 1]
            NB = data[:, 2]
            NC = data[:, 3]
            if switch_direction == 'A-B':
                switching_index = np.where(NB / N - switch_threshold >= -1e-7)[0]
            elif switch_direction == 'B-A':
                switching_index = np.where(NA / N - switch_threshold >= -1e-7)[0]
            elif switch_direction == 'B-C':
                switching_index = np.where(NC / N - switch_threshold >= -1e-7)[0]


            if len(switching_index):
                T_switching.append(t[switching_index[0]] / N )
                #T_switching.append(t[switching_index[0]] )
        if len(T_switching) < len(seed_list):
            print('simulation time is not enough', len(T_switching))

        T_list[i_N] = np.mean(T_switching)
    #plt.semilogy(N_list, T_list, 'o-', label=f'$P_B={pB}$')
    #plt.semilogy(N_list, T_list, 'o-', label=f'$m={number_opinion}$')
    plt.semilogx(N_list, T_list, 'o-', label=f'$m={number_opinion}$')
    #plt.semilogy(N_list, T_list, 'o-')
    plt.rc('font', size=ticksize)
    plt.xlabel('$N$', fontsize=fontsize)
    plt.ylabel('$T_{\\mathrm{switching}}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1)
    #plt.show()

    return T_switching

def evolution_xA_t(number_opinion, N_list, seed_list, pA, p0, switch_direction, approx_integer):
    """TODO: Docstring for switching_time.

    :arg1: TODO
    :returns: TODO

    """
    T_list = np.zeros((len(N_list)))
    for i_N, N in enumerate(N_list):
        des = f'../data/actual_simulation/number_opinion={number_opinion}/approx_integer=' + approx_integer + f'/N={N}_pA={pA}_p0={p0}_switch_direction=' + switch_direction + '/'
        xA_list = []
        for seed in seed_list:
            des_file = des + f'seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            t = data[:, 0]
            NA = data[:, 1]
            NB = data[:, 2]
            NC = data[:, 3]
            xA = NA / N
            t = t / N
            xA_list.append(xA)
            plt.plot(t, xA, color='tab:blue')

    plt.rc('font', size=ticksize)
    plt.xlabel('$t$', fontsize=fontsize)
    plt.ylabel('$x_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1)
    #plt.show()

    return None

def switching_time_distribution(number_opinion, N_list, interaction_number, interval, seed_list, pA, pB, switch_direction, switch_threshold, approx_integer):
    """TODO: Docstring for switching_time.

    :arg1: TODO
    :returns: TODO

    """
    #des = f'../data/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_pB={pB}_switch_direction=' + switch_direction + '/'
    for N in N_list:
        des = f'../data/actual_simulation/number_opinion={number_opinion}/approx_integer=' + approx_integer + f'/N={N}_pA={pA}_pB={pB}_switch_direction=' + switch_direction + '/'
        T_switching = []
        for seed in seed_list:
            des_file = des + f'seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            t = data[:, 0]
            NA = data[:, 1]
            NB = data[:, 2]
            if switch_direction == 'A-B':
                switching_index = np.where(NB / N - switch_threshold > -1e-5)[0]
            elif switch_direction == 'B-A':
                switching_index = np.where(NA / N - switch_threshold > -1e-5)[0]
            if len(switching_index):
                T_switching.append(t[switching_index[0]] / N )
        T_switching = np.array(T_switching)
        bins = np.arange(0, np.max(T_switching), 2)
        count = np.zeros(np.size(bins))
        for i, bin_i in enumerate(bins):
            count[i] = np.sum(T_switching > bin_i) / len(seed_list)
        plt.semilogy(bins, count, '.-', label=f'N={N}')
    plt.rc('font', size=ticksize)
    plt.xlabel('$t$', fontsize=fontsize)
    plt.ylabel('$P_{not}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)

    plt.legend(frameon=False, fontsize = legendsize, markerscale=3)
    plt.show()
    return T_switching

    


number_opinion = 22
N = 10000
interaction_number = N * 100
data_point = 1000
interval = int(interaction_number/data_point)
seed_list = np.arange(100)
pA = 0.2
p = 0.01
N_list = np.array([1000, 10000, 100000, 1000000])
interaction_number_list = N_list * 100
for N, interaction_number in zip(N_list, interaction_number_list):
    #plot_actual_simulation(number_opinion, N, interaction_number, interval, seed_list, pA, p, 'C')
    #compare_actual_mfe(number_opinion, N, interaction_number, interval, seed_list, pA, p, 'C')
    pass
N_list = np.array([1000, 5000, 10000, 50000, 100000])
interaction_number_list = N_list * 100
p_list = np.array([0.01, 0.02, 0.03, 0.04])
pA = 0.18
for p in p_list:
    #m, b = consensus_time_size(number_opinion, N_list, interaction_number_list, data_point, seed_list, pA, p)
    pass
N_list = np.array([1000])
interaction_number_list = N_list * 100
M_list = [12]
#consensus_time_slope(M_list)
#plt.show()



number_opinion = 2
N = 100000
interaction_number = N * 100
data_point = 1000
interval = int(interaction_number / data_point)
seed_list = np.arange(50)
pA = 0
p = 0
#plot_actual_simulation(number_opinion, N, interaction_number, interval, seed_list, pA, p, 'A')
number_opinion = 2
N_list = np.array([100, 1000, 10000, 100000, 1000000])
interaction_number_list = N_list * 100
seed_list = np.arange(100)
pA = 0.12
data_point = 1000
#compare_actual_mfe_onecommitted(number_opinion, N_list, interaction_number_list, data_point, seed_list, pA)
#Tc_N_onecommitted(number_opinion, N_list, interaction_number_list, seed_list, pA)
number_opinion_list = [4, 5, 6, 7, 8]
N = 1000
interaction_number = N * 100
pA = 0.1
p_list = [0.05, 0.0333, 0.025, 0.02, 0.0166]
for number_opinion, p in zip(number_opinion_list, p_list):
    #Tc_m_actual(number_opinion, N, interaction_number, seed_list, pA, p)
    #Tc_m_mft(number_opinion, pA, p)
    pass
number_opinion = 2
N = 100
interaction_number = N * 10000
interval = int(interaction_number / data_point)
seed_list = [0]
pA = 0.3
pB = 0
xA = 0.04
switch_direction = 'B-A'
plot_opinion = 'A'
#plot_actual_simulation_two_opinion(number_opinion, N, interaction_number, interval, seed_list, pA, pB, xA, plot_opinion)


seed_list = np.arange(400) 
switch_threshold = 1 - pA
switch_direction = 'B-A'
approx_integer = 'floor'
N_list = np.array([60, 80, 100, 120, 140, 160, 180, 200, 240, 260, 280])
N_list = np.array([60, 80, 100, 120, 140, 160, 180, 200, 220, 240])
N_list = np.array([50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500])
N_list = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
N_list = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 6000, 7000, 8000, 9000, 10000])

N_list = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000])

switching_time_N(number_opinion, N_list, interaction_number, interval, seed_list, pA, pB, switch_direction, switch_threshold, approx_integer)
#switching_time_distribution(number_opinion, N_list, interaction_number, interval, seed_list, pA, pB, switch_direction, switch_threshold, approx_integer)

seed_list = np.arange(400)
number_opinion = 3
N_list = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900])
N_list = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400])
N_list = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])
pA = 0.3
p0 = 0.08
switch_direction = 'B-A'

switch_threshold = 0.7352795
switch_threshold = 0.7322767
switch_threshold = 0.72566165
switch_threshold = 0.7088873

switch_threshold = 0.574541378
switch_threshold = 0.572979745
switch_threshold = 0.5697201
switch_threshold = 0.56254679
approx_integer = 'floor'
#switching_time_N_multi(number_opinion, N_list, seed_list, pA, p0, switch_direction, switch_threshold, approx_integer)
N_list = [10000]
#evolution_xA_t(number_opinion, N_list, seed_list, pA, p0, switch_direction, approx_integer)
