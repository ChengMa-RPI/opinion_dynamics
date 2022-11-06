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

mpl.rcParams['axes.prop_cycle'] = (cycler(color=[i for i in ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'] for j in range(1)] ))
#mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:red', 'tab:olive', 'tab:cyan']) 




number_opinion = 3
fontsize = 22
ticksize= 15
legendsize = 15
alpha = 0.8
lw = 3

def plot_actual_simulation_BKS(number_opinion, N, interaction_number, interval, seed_list, pA, p, plot_opinion):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    des = f'../data/approximation_BKS/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_p={p}/'
    NA_list = []
    NB_list = []
    NC_list = []
    ND_list = []

    for seed in seed_list:
        des_file = des + f'seed={seed}_full.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        t = data[:, 0]
        NA = data[:, 1]
        NB = data[:, 2]
        NC = data[:, 3]
        ND = data[:, 4]
        NA_list.append(NA)
        NB_list.append(NB)
        NC_list.append(NC)
        ND_list.append(ND)
    NA_list = np.vstack((NA_list)).transpose()
    NB_list = np.vstack((NB_list)).transpose()
    NC_list = np.vstack((NC_list)).transpose()
    ND_list = np.vstack((ND_list)).transpose()
    xA_list = NA_list/N
    xB_list = NB_list/N
    xC_list = NC_list/N
    xD_list = ND_list/N
    if plot_opinion == 'A':
        y = xA_list
        ylabels = '$x_0$'
    if plot_opinion == 'B':
        y = xB_list
        ylabels = '$x_1$'
    if plot_opinion == 'C':
        y = xC_list
        ylabels = '$x_2$'
    if plot_opinion == 'D':
        y = xD_list
        ylabels = '$x_3$'
    plt.plot(t/N, y[:, :1], color='tab:green', alpha=alpha, label='ABM one')
    plt.plot(t/N, np.mean(y, 1), color='tab:blue', alpha=alpha, linewidth=lw, label='ABM average')
    plt.ylabel(ylabels, fontsize=fontsize)
    #save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xA.png'
    plt.rc('font', size=ticksize)
    #plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.xlabel('$t$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params(nbins=6)
    #plt.savefig(save_des)
    #plt.close('all')

    return None

def plot_actual_simulation_BKS_smallcommitted(number_opinion, N, interaction_number, interval, seed_list, pA, p, plot_ave):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    des = f'../data/approximation_BKS/actual_simulation/number_opinion={number_opinion}/N={N}_interaction_number={interaction_number}_pA={pA}_p={p}/'
    NA_list = []
    NB_list = []
    NC_list = []
    ND_list = []

    for seed in seed_list:
        des_file = des + f'seed={seed}_full.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        t = data[:, 0]
        NA = data[:, 1]
        NB = data[:, 2]
        NC = data[:, 3]
        ND = data[:, 4]
        NA_list.append(NA)
        NB_list.append(NB)
        NC_list.append(NC)
        ND_list.append(ND)
    NA_list = np.vstack((NA_list)).transpose()
    NB_list = np.vstack((NB_list)).transpose()
    NC_list = np.vstack((NC_list)).transpose()
    ND_list = np.vstack((ND_list)).transpose()
    xA_list = NA_list/N
    xB_list = NB_list/N
    xC_list = NC_list/N
    xD_list = ND_list/N
    if plot_ave:
        plt.plot(t/N, np.mean(xB_list, 1), color='tab:green', alpha=alpha, linewidth=lw, label='$o_1$')
        plt.plot(t/N, np.mean(xC_list, 1), color='tab:blue', alpha=alpha, linewidth=lw, label='$o_2$')
        plt.plot(t/N, np.mean(xD_list, 1), color='tab:red', alpha=alpha, linewidth=lw, label='$o_3$')
    else:
        plt.plot(t/N, xB_list[:, :1], color='tab:green', alpha=alpha, label='$o_1$')
        plt.plot(t/N, xC_list[:, :1], color='tab:blue', alpha=alpha, label='$o_2$')
        plt.plot(t/N, xD_list[:, :1], color='tab:red', alpha=alpha, label='$o_3$')
    plt.ylabel('$x$', fontsize=fontsize)
    #save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xA.png'
    plt.rc('font', size=ticksize)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('$t$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params(nbins=6)
    #plt.savefig(save_des)
    #plt.close('all')

    return None

def plot_mfode_BKS(number_opinion, pA, p, plot_opinion):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    des = f'../data/approximation_BKS/mfe/number_opinion={number_opinion}/'

    des_file = des + f'pA={pA}_p={p}.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    t = data[:, 0]
    xA = data[:, 1]
    xB = data[:, 2]
    xC = data[:, 3]
    xD = data[:, 4]
    if plot_opinion == 'A':
        y = xA
        ylabels = '$x_0$'
    elif plot_opinion == 'B':
        y = xB
        ylabels = '$x_1$'
    elif plot_opinion == 'C':
        y = xC
        ylabels = '$x_2$'
    elif plot_opinion == 'D':
        y = xD
        ylabels = '$x_3$'
    plt.plot(t, y, color='tab:red', alpha=alpha, linewidth=lw, label='MF-ODE')
    plt.ylabel(ylabels, fontsize=fontsize)
    #save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xA.png'
    plt.rc('font', size=ticksize)
    #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('$t$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    #plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, 0.))
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params(nbins=6)
    #plt.savefig(save_des)
    #plt.close('all')

    return None

def plot_mfode_mixed_BKS(number_opinion, pA, p, plot_opinion):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    des = f'../data/approximation_BKS/mfe/number_opinion={number_opinion}/'

    des_file = des + f'pA={pA}_p={p}.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    t = data[:, 0]
    xAB = data[:, 9]
    xAC = data[:, 10]
    xAD = data[:, 11]
    xBC = data[:, 12]
    xBD = data[:, 13]
    xCD = data[:, 14]
    xABC = data[:, 15]
    xABD = data[:, 16]
    xACD = data[:, 17]
    xBCD = data[:, 18]
    xABCD = data[:, 19]
    if plot_opinion == 'AB':
        y = xAB
        labels = '$x_{01}$'
    elif plot_opinion == 'AC':
        y = xAC
        labels = '$x_{02}$'
    elif plot_opinion == 'AD':
        y = xAD
        labels = '$x_{03}$'
    elif plot_opinion == 'BC':
        y = xBC
        labels = '$x_{12}$'
    elif plot_opinion == 'BD':
        y = xBD
        labels = '$x_{13}$'
    elif plot_opinion == 'CD':
        y = xCD
        labels = '$x_{23}$'
    elif plot_opinion == 'ABC':
        y = xABC
        labels = '$x_{012}$'
    elif plot_opinion == 'ABD':
        y = xABD
        labels = '$x_{013}$'
    elif plot_opinion == 'ACD':
        y = xACD
        labels = '$x_023$'
    elif plot_opinion == 'BCD':
        y = xBCD
        labels = '$x_{123}$'
    elif plot_opinion == 'ABCD':
        y = xABCD
        labels = '$x_{0123}$'



    plt.semilogy(t, y, alpha=alpha, linewidth=lw, label=labels)
    plt.ylabel('$x$', fontsize=fontsize)
    #save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xA.png'
    plt.rc('font', size=ticksize)
    #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('$t$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    #plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, 0.))
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params('x', nbins=6)
    #plt.savefig(save_des)
    #plt.close('all')

    return None

def plot_mfode_single_BKS(number_opinion, pA, p, plot_type):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    des = f'../data/approximation_BKS/mfe_LO/number_opinion={number_opinion}/'

    des_file = des + f'pA={pA}_p={p}.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    t = data[:, 0]
    xA = data[:, 1]
    xB = data[:, 2]
    xC = data[:, 3]
    xD = data[:, 4]

    if plot_type == 'square':
        plt.plot(t, xA/pA**2, color='tab:red', alpha=alpha, linewidth=lw, label='$0$')
        plt.plot(t, xB/p**2, color='tab:blue', alpha=alpha, linewidth=lw, label='$1$')
        plt.ylabel('$x/c^2$', fontsize=fontsize)
    else:
        plt.semilogy(t, xA, color='tab:red', alpha=alpha, linewidth=lw, label='$0$')
        plt.semilogy(t, xB, color='tab:blue', alpha=alpha, linewidth=lw, label='$1$')
        plt.ylabel('$x$', fontsize=fontsize)

    #save_des = f'../report/report042521/number_opinion={number_opinion}_N={N}_pA={pA}_p={p}_xA.png'
    plt.rc('font', size=ticksize)
    #plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('$t$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    #plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, 0.))
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params('x', nbins=6)
    save_des = f'../report/report082121/BKS_number_opinion={number_opinion}_pA={pA}_p={p}_' + plot_type + '.png'
    plt.savefig(save_des)
    plt.close('all')

    return None



number_opinion = 4
N = 1000000
interaction_number = N * 100
data_point = 1000
interval = int(interaction_number / data_point)
seed_list = np.arange(100).tolist()
pA = 0.1
p = 0.01
plot_opinion = 'A'
#plot_mfode_BKS(number_opinion, pA, p, plot_opinion)
#plot_actual_simulation_BKS(number_opinion, N, interaction_number, interval, seed_list, pA, p, plot_opinion)
plot_ave = 0
#plot_actual_simulation_BKS_smallcommitted(number_opinion, N, interaction_number, interval, seed_list, pA, p, plot_ave)
plot_opinion_list = ['AB', 'BC', 'ABC', 'BCD', 'ABCD']
for plot_opinion in plot_opinion_list:
    #plot_mfode_mixed_BKS(number_opinion, pA, p, plot_opinion)
    pass
number_opinion = 2
pA = 0.1
p = 0.01
number_opinion_list = [2, 3, 4]
plot_type_list = ['square', 'direct']
for number_opinion in number_opinion_list:
    for plot_type in plot_type_list:
        plot_mfode_single_BKS(number_opinion, pA, p, plot_type)
