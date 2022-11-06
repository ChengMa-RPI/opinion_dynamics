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

mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:olive', 'tab:cyan']) 
mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']) )

number_opinion = 3
fontsize = 22
ticksize= 15
legendsize = 15
alpha_color = 0.8
lw = 3

def pA_critical_N(N_list, pAtilde, ode):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    pA_c_list = []
    if ode:
        des = '../data/listener_only/ode/pA_critical_N/'
    else:
        des = '../data/listener_only/recursive/pA_critical_N/'
    for N in N_list:
        des_file = des + f'pAtilde={pAtilde}_N={N}.csv'

        data = np.array(pd.read_csv(des_file, header=None))
        p_list = data[:, :N]
        pA = data[:, 0]
        xA_fraction = data[:, N] / np.sum(data[:, N:N*2], 1)
        index_sort = np.argsort(pA)
        xA_fraction_sort = xA_fraction[index_sort]
        dominate_index = np.where(xA_fraction_sort > 0.5)[0][0]
        pA_c = pA[index_sort][dominate_index]
        pA_c_list.append(pA_c)
        labels = '$P_\\tilde{A}=' + f'{pAtilde}$'
        labels = '$S_1$'
        linestyle = '--'
    plt.plot(N_list, pA_c_list, linestyle, linewidth=lw, alpha=alpha_color, label=labels, color= '#fc8d62')
    #plt.plot(N_list, pA_c_list, linestyle, linewidth=lw, alpha=alpha_color, label=labels)
    plt.xlabel('$m$', fontsize=fontsize)
    plt.ylabel('$P_A^{(c)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.5)
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def pA_critical_N_fluctuation(N_list, pAtilde, max_fluctuation, seed_list, ode):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    pA_c_list = np.zeros((np.size(N_list), np.size(seed_list)))
    if ode:
        des = '../data/listener_only/ode/pA_critical_N_fluctuation/'
    else:
        des = '../data/listener_only/recursive/pA_critical_N_fluctuation/'
    for i, N in enumerate(N_list):
        for j, seed in enumerate(seed_list):
            des_file = des + f'pAtilde={pAtilde}_N={N}_maxfluc={max_fluctuation}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            p_list = data[:, :N]
            pA = data[:, 0]
            xA_fraction = data[:, N] / np.sum(data[:, N:N*2], 1)
            index_sort = np.argsort(pA)
            xA_fraction_sort = xA_fraction[index_sort]
            dominate_index = np.where(xA_fraction_sort > 0.5)[0][0]
            pA_c = pA[index_sort][dominate_index]
            pA_c_list[i, j] = pA_c
    #labels = '$P_\\tilde{A}=' + f'{pAtilde}$'
    if ode:
        labels = 'ODE'
        linestyle = '--'
    else:
        labels = 'recursive'
        linestyle = '-.'
    plt.plot(N_list, pA_c_list, linestyle, marker='o', linewidth=lw, alpha=alpha_color, label=labels)
    plt.xlabel('$N$', fontsize=fontsize)
    plt.ylabel('$P_A^{(c)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.5)
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def pA_critical_N_fluctuation_dirichlet(N_list, pAtilde, alpha, seed_list, ode):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    if ode:
        des = '../data/listener_only/ode/pA_critical_N_fluctuation_dirichlet/'
    else:
        des = '../data/listener_only/recursive/pA_critical_N_fluctuation_dirichlet/'
    for i, N in enumerate(N_list):
        pA_c_list = []
        for j, seed in enumerate(seed_list):
            des_file = des + f'pAtilde={pAtilde}_N={N}_alpha={alpha}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            p_list = data[:, :N]
            pA = data[:, 0]
            p_max = np.max(p_list[:, 1:])
            xA_fraction = data[:, N] / np.sum(data[:, N:N*2], 1)
            index_sort = np.argsort(pA)
            xA_fraction_sort = xA_fraction[index_sort]
            dominate_index = np.where(xA_fraction_sort > 0.5)[0][0]
            pA_c = pA[index_sort][dominate_index]
            if pA_c - p_max >2e-3:
                pA_c_list.append(pA_c)
        if i < np.size(N_list)-1:
            plt.plot(N * np.ones(np.size(pA_c_list)), pA_c_list, 'o', alpha=alpha_color, color='#66c2a5')
        else:
            plt.plot(N * np.ones(np.size(pA_c_list)), pA_c_list, 'o', alpha=alpha_color, color='#66c2a5', label='$S_0$')
    plt.xlabel('$m$', fontsize=fontsize)
    plt.ylabel('$P_A^{(c)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.5)
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def pA_critical_N_lowerbound(N_list, pAtilde, ode):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    pA_c_list = np.zeros((np.size(N_list)))
    if ode:
        des = '../data/listener_only/ode/pA_critical_N_lowerbound/'
    else:
        des = '../data/listener_only/recursive/pA_critical_N_lowerbound/'
    for i, N in enumerate(N_list):
        des_file = des + f'pAtilde={pAtilde}_N={N}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        p_list = data[:, :4]
        pA = data[:, 0]
        number_Ctilde = np.round((pAtilde - p_list[:, 2]) / p_list[:, 3] )
        xA_fraction = data[:, 4] / (data[:, 4] + data[:, 5] + data[:, 6] + data[:, 7] * number_Ctilde)
        index_sort = np.argsort(pA)
        xA_fraction_sort = xA_fraction[index_sort]
        dominate_index = np.where(xA_fraction_sort > 0.5)[0][0]
        pA_c = pA[index_sort][dominate_index]
        pA_c_list[i] = pA_c
        #labels = '$P_\\tilde{A}=' + f'{pAtilde}$'
        labels = 'S2'
        linestyle = '--'
    plt.plot(N_list, pA_c_list, linestyle, linewidth=lw, alpha=alpha_color, label=labels, color='#8da0cb')
    plt.xlabel('$m$', fontsize=fontsize)
    plt.ylabel('$P_A^{(c)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.5)
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def pA_critical_pstd_fluctuation_dirichlet(N_list, pAtilde, alpha, seed_list, ode):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    if ode:
        des = '../data/listener_only/ode/pA_critical_N_fluctuation_dirichlet/'
    else:
        des = '../data/listener_only/recursive/pA_critical_N_fluctuation_dirichlet/'
    for i, N in enumerate(N_list):
        pA_c_list = []
        p_std = []
        for j, seed in enumerate(seed_list):
            des_file = des + f'pAtilde={pAtilde}_N={N}_alpha={alpha}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            p_list = data[:, :N]
            pA = data[:, 0]
            p_max = np.max(p_list[:, 1:])
            xA_fraction = data[:, N] / np.sum(data[:, N:N*2], 1)
            index_sort = np.argsort(pA)
            xA_fraction_sort = xA_fraction[index_sort]
            dominate_index = np.where(xA_fraction_sort > 0.5)[0][0]
            pA_c = pA[index_sort][dominate_index]
            if pA_c - p_max >2e-3:
                pA_c_list.append(pA_c)
                p_std.append(np.std(p_list[0, 1:]))
        plt.plot(p_std, pA_c_list, 'o', alpha=alpha_color, label=f'$m={N}$')
    plt.xlabel('$std(P_i)$', fontsize=fontsize)
    plt.ylabel('$P_A^{(c)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.5)
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def nA_PA(N_list, pAtilde, ode):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    pA_c_list = []
    if ode:
        des = '../data/listener_only/ode/pA_critical_N/'
    else:
        des = '../data/listener_only/recursive/pA_critical_N/'
    for N in N_list:
        des_file = des + f'pAtilde={pAtilde}_N={N}.csv'

        data = np.array(pd.read_csv(des_file, header=None))
        p_list = data[:, :N]
        pA = data[:, 0]
        xA = data[:, N]
        nA = xA + pA
        valid_index = np.where(pA < 0.1)[0]
        index_sort = np.argsort(pA[valid_index])
        linestyle = '--'
        labels = f'$m={N}$'
        labels = f'$S_1$'
        plt.plot(pA[valid_index][index_sort], nA[valid_index][index_sort], linestyle, linewidth=lw, alpha=alpha_color, label=labels, color='#fc8d62')
    plt.xlabel('$P_A$', fontsize=fontsize)
    plt.ylabel('$n_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    #plt.xlim(-0.005, 0.105)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=3)
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def nA_PA_lowerbound(N_list, pAtilde, ode):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    pA_c_list = np.zeros((np.size(N_list)))
    if ode:
        des = '../data/listener_only/ode/pA_critical_N_lowerbound/'
    else:
        des = '../data/listener_only/recursive/pA_critical_N_lowerbound/'
    for i, N in enumerate(N_list):
        des_file = des + f'pAtilde={pAtilde}_N={N}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        p_list = data[:, :4]
        pA = data[:, 0]
        xA = data[:, 4]
        nA = xA + pA
        valid_index = np.where(pA < 0.1)[0]
        index_sort = np.argsort(pA[valid_index])
        labels = '$S_2$'
        linestyle = '--'
        plt.plot(pA[valid_index][index_sort], nA[valid_index][index_sort], linestyle, linewidth=lw, alpha=alpha_color, label=labels, color='#8da0cb')
    plt.xlabel('$P_A$', fontsize=fontsize)
    plt.ylabel('$n_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=3)
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def nA_PA_fluctuation_dirichlet(N_list, pAtilde, alpha, seed_list, ode):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    if ode:
        des = '../data/listener_only/ode/pA_critical_N_fluctuation_dirichlet/'
    else:
        des = '../data/listener_only/recursive/pA_critical_N_fluctuation_dirichlet/'
    for i, N in enumerate(N_list):
        pA_list = []
        xA_list = []
        for j, seed in enumerate(seed_list):
            des_file = des + f'pAtilde={pAtilde}_N={N}_alpha={alpha}_seed={seed}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            p_list = data[:, :N]
            p_max = np.max(p_list[:, 1:], 1)
            pA = data[:, 0]
            valid_index = np.where(pA - p_max > 2e-3)[0]
            valid_index = np.where((pA - p_max > 2e-3) & (pA < 0.1))[0]
            xA = data[:, N]
            pA_list.append(pA[valid_index])
            xA_list.append(xA[valid_index])
        pA_list = np.hstack((pA_list))
        xA_list = np.hstack((xA_list))
        nA_list = pA_list + xA_list
        labels = '$S_0$'
        plt.plot(pA_list, nA_list, 'o', markersize=3, alpha=alpha_color, color='#66c2a5', label=labels)
        
    plt.xlabel('$P_A$', fontsize=fontsize)
    plt.ylabel('$n_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=10)
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None


N_list = [6]
N_list = np.arange(3, 8, 1)
pAtilde = 0.2
pAtilde_list = [0.1, 0.12, 0.14, 0.16]
pAtilde_list = [0.18]
ode = 1
max_fluctuation = 0.01
seed_list = np.arange(30).tolist()
alpha = 0.5
alpha = 10
for pAtilde in pAtilde_list:
    #pA_critical_N_fluctuation(N_list, pAtilde, max_fluctuation, seed_list, ode)
    pA_critical_N_fluctuation_dirichlet(N_list, pAtilde, alpha, seed_list, ode)
    pA_critical_N(N_list, pAtilde, ode)
    pA_critical_N_lowerbound(N_list, pAtilde, ode)
    #pA_critical_pstd_fluctuation_dirichlet(N_list, pAtilde, alpha, seed_list, ode)
    #nA_PA_fluctuation_dirichlet(N_list, pAtilde, alpha, seed_list, ode)
    #nA_PA(N_list, pAtilde, ode)
    #nA_PA_lowerbound(N_list, pAtilde, ode)
