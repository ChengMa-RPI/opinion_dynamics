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

mpl.rcParams['axes.prop_cycle'] = (cycler(color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']))

number_opinion = 3
fontsize = 22
ticksize= 15
legendsize = 15
alpha = 0.8
lw = 3

def compare_original_first_order(number_opinion, p, error_list):
    """TODO: Docstring for compare_original_first_order.

    :number_opinion: TODO
    :p: TODO
    :returns: TODO

    """
    des_original = f'../data/num_opinion={number_opinion}_original/p={p}_random.csv'
    des_approx = f'../data/num_opinion={number_opinion}_first_order/p={p}_random.csv'
    des_reduce = f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_three/p0={p}.csv'
    data_original = np.array(pd.read_csv(des_original, header=None))
    data_approx = np.array(pd.read_csv(des_approx, header=None))
    data_reduce = np.array(pd.read_csv(des_reduce, header=None))
    committed_approx = data_approx[:, :number_opinion]
    xA_approx = data_approx[:, -3]
    committed_original = data_original[:, :number_opinion]
    index = np.where(committed_original[:, 0] - p>1e-4)[0]
    committed_original = committed_original[index]
    xA_original = data_original[:, -number_opinion]
    xA_original = xA_original[index]
    committed_reduce = data_reduce[:, 0]
    xA_reduce = data_reduce[:, -3]
    index_approx = []
    for committed_original_i in committed_original:
        index = np.where(np.sum(abs(committed_approx - committed_original_i)<1e-5, 1) == number_opinion)[0]
        index_approx.append(index)
    index_reduce = []
    for committed_original_i in committed_original:
        index = np.where(abs(committed_reduce - committed_original_i[0])<6e-4)[0]
        index_reduce.append(index)
    index_approx = np.hstack((index_approx))
    index_reduce = np.hstack((index_reduce))
    xA_approx_sort = xA_approx[index_approx]
    xA_reduce_sort = xA_reduce[index_reduce]
    #plt.plot(xA_original, xA_approx_sort, '.')
    #plt.plot(xA_original, xA_reduce_sort, '.')
    n1_list = []
    n2_list = []
    for error in error_list:
        n1 = np.sum(np.abs(xA_original - xA_approx_sort)/xA_original < error)/np.size(xA_original)
        n2 = np.sum(np.abs(xA_original - xA_reduce_sort)/xA_original < error)/np.size(xA_original)
        n1_list.append(n1)
        n2_list.append(n2)
    plt.plot(error_list, n2_list, linewidth=lw, alpha=alpha, label='$S_1$')
    plt.plot(error_list, n1_list, linewidth=lw, alpha=alpha, label='$S_3$')

    plt.subplots_adjust(left=0.20, right=0.95, wspace=0.25, hspace=0.25, bottom=0.15, top=0.95)
    plt.xlabel('error', fontsize=fontsize)
    plt.ylabel('accuracy', fontsize=fontsize)

    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize)
    plt.show()

    return None

number_opinion = 5
p = 0.01
error_list = np.arange(0.01, 0.3, 0.01)
compare_original_first_order(number_opinion, p, error_list)
