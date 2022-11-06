import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl
import itertools


mpl.rcParams['axes.prop_cycle'] = cycler(color=['tab:blue', 'tab:green', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'grey', 'tab:red', 'tab:olive', 'tab:cyan']) 
mpl.rcParams['axes.prop_cycle'] = (cycler(color=[i for i in ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3'] for j in range(1)] ))




number_opinion = 3
fontsize = 22
ticksize= 15
legendsize = 15
alpha = 0.8
lw = 3



def nA_pA(number_opinion, p_cAtilde):
    """TODO: Docstring for p_cA_p_cAtilde.

    :number_opinion: TODO
    :p_cAtilde: TODO
    :returns: TODO

    """
    des = '../data/big_small_committed/'
    des_file = des + f'num_opinion={number_opinion}_p_cAtilde={p_cAtilde}.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    pA = data[:, 0]
    index_sort_pA = np.argsort(pA)
    pA_sort = pA[index_sort_pA]

    p_Atilde = data[index_sort_pA, 1]
    xA = data[index_sort_pA, 2]
    x_Atilde = data[index_sort_pA, 4]
    #plt.plot(pA_sort, xA + pA_sort, label='$p_{\\tilde{A}}=$' + f'{p_Atilde[0]}')
    plt.plot(pA_sort[:], xA[:] + pA_sort[:], label='$m=$' + f'{number_opinion-1}')
    plt.rc('font', size=ticksize)
    plt.xlabel('$P_A$', fontsize=fontsize)
    plt.ylabel('$n_{A}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=3)
    #plt.show()

    return None

def pA_c_pAtilde(number_opinion, p_cAtilde_list):
    """TODO: Docstring for p_cA_p_cAtilde.

    :number_opinion: TODO
    :p_cAtilde: TODO
    :returns: TODO

    """
    des = '../data/big_small_committed/'
    pA_transition_list = []
    for p_cAtilde in p_cAtilde_list:
        print(p_cAtilde)
        des_file = des + f'num_opinion={number_opinion}_p_cAtilde={p_cAtilde}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        pA = data[:, 0]
        index_sort_pA = np.argsort(pA)
        pA_sort = pA[index_sort_pA]

        p_Atilde = data[index_sort_pA, 1]
        nA = data[index_sort_pA, 2] + pA_sort
        index_transition = np.where(nA > 0.5)[0][0]
        pA_transition = pA_sort[index_transition]
        pA_transition_list.append(pA_transition)
    plt.plot(p_cAtilde_list, pA_transition_list, label='$m=$' + f'{number_opinion-1}')
    plt.rc('font', size=ticksize)
    plt.xlabel('$P_{\\tilde{A}}$', fontsize=fontsize)
    plt.ylabel('$P_{A}^{c}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.92, wspace=0.25, hspace=0.25, bottom=0.15, top=0.92)

    plt.legend(frameon=False, fontsize = legendsize, markerscale=3)

    #plt.show()
    return None



number_opinion = 3
p_cAtilde_list = np.round(np.arange(0, 0.2, 0.01), 2)
for p_cAtilde in p_cAtilde_list:
    #nA_pA(number_opinion, p_cAtilde)
    pass

p_cAtilde = 0.06
number_opinion_list = [3, 4, 5, 6, 7, 8]
for number_opinion in number_opinion_list:
    nA_pA(number_opinion, p_cAtilde)
    pass

for number_opinion in number_opinion_list:
    #pA_c_pAtilde(number_opinion, p_cAtilde_list)
    pass
