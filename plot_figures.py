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


number_opinion = 3
fontsize = 22
ticksize= 15
legendsize = 15
alpha = 0.8
lw = 3

def pqr_committed(number_opinion):
    """TODO: Docstring for number_committed.

    :number_opinion: TODO
    :: TODO
    :returns: TODO

    """
    data = pd.read_csv(f'../data/num_opinion={number_opinion}_stable.csv', header=None, sep='\n')
    region_separate = [[] for _ in range(6)]
    for i in data[0]:
        line = [float(x) for x in i.split(',')]
        committed_fraction = line[:number_opinion]
        stable_state = line[number_opinion:]
        stable_state = np.array(stable_state).reshape(number_opinion, int(len(stable_state)/number_opinion))
        number_stable = int(np.size(stable_state) / number_opinion)
        region_separate[number_stable-1].append(committed_fraction)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for region, i in zip(region_separate, range(len(region_separate))):
        region = np.vstack((region))
        ax.scatter3D(region[:, 0], region[:, 1], region[:, 2], label=f'{i+1}')
    ax.invert_zaxis()
    ax.invert_xaxis()
    ax.set_xlabel('$p_A$', fontsize=fontsize)
    ax.set_ylabel('$p_B$', fontsize=fontsize)
    ax.set_zlabel('$p_C$', fontsize=fontsize)
    labels=[0.01, 0, 0.05, 0.1, 0.15]
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_yticklabels([str(label) for label in labels])
    ax.set_zticklabels([str(label) for label in labels])
    ax.xaxis.set_tick_params(labelsize=ticksize, rotation=0, pad=0)
    ax.yaxis.set_tick_params(labelsize=ticksize, pad=0)
    ax.zaxis.set_tick_params(labelsize=ticksize, rotation=-0, pad=2)
    plt.subplots_adjust(left=0.18, right=0.82, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, loc='lower right', bbox_to_anchor=(1.25, 0.))
    plt.locator_params(nbins=5)
    plt.show()

def cross_section_committed(number_opinion, p_fix):
    """TODO: Docstring for cross_section_committed.

    :arg1: TODO
    :returns: TODO

    """
    region_separate = [[] for _ in range(6)]
    #data = pd.read_csv(f'../data/num_opinion={number_opinion}_stable.csv', header=None, sep='\n')
    data = pd.read_csv(f'../data/num_opinion={number_opinion}_strongstable.csv', header=None, sep='\n')

    for i in data[0]:
        line = [float(x) for x in i.split(',')]
        committed_fraction = line[:number_opinion]
        stable_state = line[number_opinion:]
        stable_state = np.array(stable_state).reshape(number_opinion, int(len(stable_state)/number_opinion))
        number_stable = int(np.size(stable_state) / number_opinion)
        region_separate[number_stable-1].append(committed_fraction)

    for region, i in zip(region_separate, range(len(region_separate))):
        if region:
            print(i)
            region = np.vstack((region))
            p = region[:, 0]
            q = region[:, 1]
            r = region[:, 2]
            index = np.where(p == p_fix)[0]
            if len(index):
                plt.plot(q[index], r[index], 'o', alpha=alpha, markersize=10, label=f'{i+1}')
    plt.axis('square')
    plt.xlabel('$p_B$', fontsize=fontsize)
    plt.ylabel('$p_C$', fontsize=fontsize)
    plt.subplots_adjust(left=0.18, right=0.82, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.legend(frameon=False, fontsize = legendsize, loc='lower right',bbox_to_anchor=(1.25, 0.))
    plt.locator_params(nbins=5)
    
    save_des = f'../report/report013121/strongstable_cross_section_p={p_fix}.png'
    plt.savefig(save_des)
    plt.close('all')
    #plt.show()
    return None

def basin_attraction_number(committed_fraction):
    """TODO: Docstring for basin_attractor.

    :arg1: TODO
    :returns: TODO

    """
    colors = ['tab:blue', 'tab:brown', 'tab:green']
    des_file = f'../data/num_opinion={number_opinion}/committed_fraction={committed_fraction}.csv' 
    #des_file = f'../data/num_opinion={number_opinion}/strong_committed_fraction={committed_fraction}.csv' 
    data = np.array(pd.read_csv(des_file, header=None))
    initial_condition = data[:, :3]
    attractors = data[:, 3:]
    attractors = np.round(attractors, 2)
    attractor_unique = np.unique(attractors, axis=0)
    for attractor, i in zip(attractor_unique[::-1], range(len(attractor_unique))):
        attractor = np.round(attractor, 2)
        index = np.where(np.sum(attractor==attractors, 1) == 3)[0]
        xy = initial_condition[index, :3]
        x = xy[:, 0]
        y = xy[:, 1]
        label = f's{i+1}'
        label = '(' + '{0:g}'.format(abs(attractor[0])) + ',' + '{0:g}'.format(abs(attractor[1])) +',' + '{0:g}'.format(abs(attractor[2])) + ')'
        if sorted(attractor)[1] == sorted(attractor)[2]:
            continue
        elif attractor[0]  == max(attractor[:3]):
            color='tab:red'
        elif attractor[1]  == max(attractor[:3]):
            color='tab:blue'
        elif attractor[2]  == max(attractor[:3]):
            color='tab:orange'
        plt.plot(x, y, '.', alpha=alpha, label=label, color=color)
    plt.axis('square')
    plt.xlabel('$x_A(0)$', fontsize=fontsize)
    plt.ylabel('$x_B(0)$', fontsize=fontsize)
    plt.subplots_adjust(left=0.10, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=4, bbox_to_anchor=(1.25, 1.))
    plt.locator_params(nbins=5)

    filename  = '[' + '{0:g}'.format(abs(committed_fraction[0])) + ', ' + '{0:g}'.format(abs(committed_fraction[1])) +', ' + '{0:g}'.format(abs(committed_fraction[2])) + ']'
    save_des = f'../figure/basin_attr_p={committed_fraction}.png'
    plt.savefig(save_des)
    #plt.close('all')
    plt.show()
    return attractor_unique

def attractor_value(number_opinion, pA_fix, pB_fix):
    """TODO: Docstring for attractor_value.

    :number_opinion: TODO
    :p_fix: TODO
    :: TODO
    :returns: TODO

    """
    region_separate = [[] for _ in range(6)]
    committed_stable = []
    #data = pd.read_csv(f'../data/num_opinion={number_opinion}_stable.csv', header=None, sep='\n')
    data = pd.read_csv(f'../data/num_opinion={number_opinion}_strongstable.csv', header=None, sep='\n')

    for i in data[0]:
        line = [float(x) for x in i.split(',')]
        committed_fraction = line[:number_opinion]
        stable_state = line[number_opinion:]
        stable_state = np.array(stable_state).reshape(int(len(stable_state)/number_opinion), number_opinion)
        stable_A = stable_state[:, 0]
        stable_B = stable_state[:, 1]
        stable_C = stable_state[:, 2]
        committed_stable.append([np.hstack((committed_fraction, xA, xB, xC)) for xA, xB, xC in zip(stable_A, stable_B, stable_C)])

    committed_stable = np.vstack((committed_stable))
    pA, pB, pC, xsA, xsB, xsC = committed_stable.transpose()
    index = np.where((pA == pA_fix) & (pB == pB_fix))[0]
    y = (xsC[index])/(1-pA_fix - pB_fix-pC[index])
    y = xsA[index]
    plt.plot(pC[index], y, 'o', alpha=alpha, label=f'$p_B={pB_fix}$')
    #plt.plot(pC[index], xsB[index], '.', color='tab:blue')
    #plt.plot(pC[index], xsA[index], '.', color='tab:green')
    plt.xlabel('$p_C$', fontsize=fontsize)
    plt.ylabel('$x_A^s$', fontsize=fontsize)
    plt.subplots_adjust(left=0.17, right=0.76, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)

    #plt.show()

    #plt.savefig(save_des)
    #plt.close('all')
    #plt.show()
    return None

def attractor_3D(number_opinion, pA_fix):
    """TODO: Docstring for attractor_value.

    :number_opinion: TODO
    :p_fix: TODO
    :: TODO
    :returns: TODO

    """
    region_separate = [[] for _ in range(6)]
    committed_stable = []
    data = pd.read_csv(f'../data/num_opinion={number_opinion}_stable.csv', header=None, sep='\n')

    for i in data[0]:
        line = [float(x) for x in i.split(',')]
        committed_fraction = line[:number_opinion]
        stable_state = line[number_opinion:]
        stable_state = np.array(stable_state).reshape(int(len(stable_state)/number_opinion), number_opinion)
        stable_A = stable_state[:, 0]
        stable_B = stable_state[:, 1]
        stable_C = stable_state[:, 2]
        committed_stable.append([np.hstack((committed_fraction, xA, xB, xC)) for xA, xB, xC in zip(stable_A, stable_B, stable_C)])

    committed_stable = np.vstack((committed_stable))
    pA, pB, pC, xsA, xsB, xsC = committed_stable.transpose()
    index = np.where((pA == pA_fix))[0]
    x = pB[index]
    y = pC[index]
    z = xsA[index]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z)
    ax.set_xlabel('$p_B$', fontsize=fontsize)
    ax.set_ylabel('$p_C$', fontsize=fontsize)
    ax.set_zlabel('$x_A^s$', fontsize=fontsize)
    labels=[0, 0.05, 0.1, 0.15]
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_yticklabels([str(label) for label in labels])
    ax.xaxis.set_tick_params(labelsize=ticksize, rotation=0, pad=0)
    ax.yaxis.set_tick_params(labelsize=ticksize, pad=0)
    ax.zaxis.set_tick_params(labelsize=ticksize, rotation=-0, pad=2)
    plt.subplots_adjust(left=0.18, right=0.82, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    #plt.legend(frameon=False, fontsize = legendsize, loc='lower right', bbox_to_anchor=(1.25, 0.))
    plt.locator_params(nbins=5)
    #plt.show()

    save_des = f'../report/report013121/attractor3D_pA={pA_fix}_xA.png'
    plt.savefig(save_des)
    plt.close('all')
    #plt.show()
    return None

def pC_critical(number_opinion):
    """TODO: Docstring for pC_critical.

    :number_opinion: TODO
    :returns: TODO

    """
    committed_stable = []
    data = pd.read_csv(f'../data/num_opinion={number_opinion}_stable.csv', header=None, sep='\n')

    for i in data[0]:
        line = [float(x) for x in i.split(',')]
        committed_fraction = line[:number_opinion]
        stable_state = line[number_opinion:]
        stable_state = np.array(stable_state).reshape(int(len(stable_state)/number_opinion), number_opinion)
        stable_A = stable_state[:, 0]
        stable_B = stable_state[:, 1]
        stable_C = stable_state[:, 2]
        committed_stable.append([np.hstack((committed_fraction, xA, xB, xC)) for xA, xB, xC in zip(stable_A, stable_B, stable_C)])

    committed_stable = np.vstack((committed_stable))
    pA, pB, pC, xsA, xsB, xsC = committed_stable.transpose()
    p_C_AB = []
    for i in range(np.size(committed_stable, 0)):
        if xsC[i] > xsA[i] and xsC[i] > xsB[i] and pA[i] > pC[i] and pB[i] > pC[i]:
            p_C_AB.append([pC[i], pA[i], pB[i]])
    p_C_AB = np.vstack((p_C_AB))
    p_AB = np.unique(p_C_AB[:, 1:], axis=0)
    p_unique = []
    for i in p_AB:
        index = np.where(np.sum(p_C_AB[:, 1:] == i, 1)==2)[0]
        p_unique.append(np.hstack((min(p_C_AB[index, 0]), i)))
    p_unique = np.vstack((p_unique))
    z, x, y = p_unique.transpose()
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z, alpha = alpha)

    ax.set_xlabel('$p_A$', fontsize=fontsize)
    ax.set_ylabel('$p_B$', fontsize=fontsize)
    ax.set_zlabel('$P_C$', fontsize=fontsize)
    labels=[0, 0, 0.05, 0.1, 0.15]
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_yticklabels([str(label) for label in labels])
    ax.xaxis.set_tick_params(labelsize=ticksize, rotation=0, pad=0)
    ax.yaxis.set_tick_params(labelsize=ticksize, pad=0)
    ax.zaxis.set_tick_params(labelsize=ticksize, rotation=-0, pad=2)
    plt.subplots_adjust(left=0.18, right=0.82, wspace=0.25, hspace=0.25, bottom=0.20, top=0.98)
    plt.locator_params(nbins=5)
    save_des = f'../report/report013121/attractor3D_pA={pA_fix}_xA.png'
    #plt.savefig(save_des)
    #plt.close('all')
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    num_colors = 10
    clrs = sns.color_palette('husl', n_colors=num_colors)
    markers = enumerate(['s', 'o', 'd', 'v'] * 4)
    i = 0
    for z_i in np.unique(z):
        z_index = np.where(z_i == z)[0]
        ax.plot(x[z_index], y[z_index], marker=next(markers)[-1], linestyle='None', color =clrs[i], label=f'$p_C={z_i}$',  markersize=10)
        i += 1
    plt.xlabel('$p_A$', fontsize=fontsize)
    plt.ylabel('$p_B$', fontsize=fontsize)
    plt.subplots_adjust(left=0.17, right=0.76, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)

    plt.show()
    #plt.contourf(x, y, z.reshape(np)
    return None

def pA_pcA_BC(number_opinion):
    """plot P_A with P_{cA} for different p_{cB}

    :number_opinion: 4
    :returns: TODO

    """
    data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_stable.csv', header=None))
    p_cA = data[:, 0]
    p_cB = data[:, 1]
    x_A = data[:, 4]
    x_B = data[:, 5]
    x_C = data[:, 6]
    x_D = data[:, 7]
    p_B_plot = np.sort(np.unique(p_cB))
    p_B_plot = np.round(np.arange(0., 0.2, 0.02), 2)
    for p in p_B_plot:
        index_p = np.where(p_cB == p)[0]
        p_cA_p = p_cA[index_p]
        x_A_p = x_A[index_p]
        x_B_p = x_B[index_p]
        x_C_p = x_C[index_p]
        x_D_p = x_D[index_p]
        index_sort = np.argsort(p_cA_p)
        y = x_B_p[index_sort] + x_C_p[index_sort] + 2 * p
        y= x_D_p[index_sort]
        y = x_A_p[index_sort] + p_cA_p[index_sort]
        plt.plot(p_cA_p[index_sort], y, '-.', linewidth=lw, alpha=alpha, label='$P_{cB}=$' + str(p))
    plt.xlabel('$P_{cA}$', fontsize=fontsize)
    plt.ylabel('$p_A^s$', fontsize=fontsize)
    #plt.subplots_adjust(left=0.17, right=0.76, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.17, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize)
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)

    plt.show()

def compare_reduce(number_opinion, p_notA):
    """plot P_A with P_{cA} for different p_{cB}

    :number_opinion: 4
    :returns: TODO

    """
    data_original = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_stable.csv', header=None))
    data_reduction = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_approximation.csv', header=None))
    p_cA1 = data_original[:, 0]
    p_cB1 = data_original[:, 1]
    x_A1 = data_original[:, 4]
    x_B1 = data_original[:, 5]
    x_C1 = data_original[:, 6]
    x_D1 = data_original[:, 7]
    index_original = np.where(p_cB1 == np.round(p_notA/2, 2))[0]
    p_cA_o = p_cA1[index_original]
    p_cB_o = p_cB1[index_original]
    x_A_o = x_A1[index_original]
    x_B_o = x_B1[index_original]
    x_C_o = x_C1[index_original]
    x_D_o = x_D1[index_original]
    y_o = x_B_o + 2 * p_cB_o + x_C_o 
    y_o = x_A_o+ p_cA_o

    p_cA2 = data_reduction[:, 0]
    p_cnotA2 = data_reduction[:, 1]
    x_A2, x_notA2, x_D2 = data_reduction[:, 2: 5].transpose()
    index_reduction = np.where(p_cnotA2 == p_notA )[0]
    p_cA_r = p_cA2[index_reduction]
    p_cnotA_r = p_cnotA2[index_reduction]
    x_A_r = x_A2[index_reduction]
    x_notA_r = x_notA2[index_reduction]
    x_D_r = x_D2[index_reduction]
    y_r = x_notA_r + p_cnotA_r
    y_r = x_A_r + p_cA_r

    plt.plot(np.sort(p_cA_r), y_r[np.argsort(p_cA_r)], '-.', linewidth=lw, alpha=alpha, label='reduction')
    plt.plot(np.sort(p_cA_o), y_o[np.argsort(p_cA_o)], '--', linewidth=lw, alpha=alpha, label='original')
    plt.xlabel('$P_{cA}$', fontsize=fontsize)
    plt.ylabel('$p_A^s$', fontsize=fontsize)
    #plt.subplots_adjust(left=0.17, right=0.76, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.17, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize)
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)

    save_des = f'../report/report020721/p_notA={p_notA}.png'
    #plt.savefig(save_des)
    #plt.close('all')

def pA_pcA_p(number_opinion, p_B_plot, plot_ABC):
    """plot P_A with P_{cA} for different p_{cB}

    :number_opinion: the number of opinions
    :returns: TODO

    """
    data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted.csv', header=None))
    p_cA = data[:, 0]
    p_cB = np.round(data[:, 1], 4)
    x_A = data[:, -number_opinion]
    x_B = data[:, -number_opinion+1]
    x_C = data[:, -number_opinion+2]
    for p in p_B_plot:
        index_p = np.where(p_cB == p)[0]
        p_cA_p = p_cA[index_p]
        p_cB_p = p_cB[index_p]
        x_A_p = x_A[index_p]
        x_B_p = x_B[index_p]
        x_C_p = x_C[index_p]
        index_sort = np.argsort(p_cA_p)
        if number_opinion == 3:
            label = '$P_C=$' + str(p)
        else:
            label = f'$p={p}$'
        if plot_ABC == 'A':
            y = x_A_p[index_sort] + p_cA_p[index_sort]
            ylabel = '$n_A$'
        elif plot_ABC == 'B':
            y = x_C_p[index_sort]
            ylabel = '$n_B$'
        elif plot_ABC == 'C':
            y = x_B_p[index_sort] + p_cB_p[index_sort]
            ylabel = '$n_C$'

        plt.plot(p_cA_p[index_sort], y, '-.', linewidth=lw, alpha=alpha, label=label)
    plt.xlabel('$P_{A}$', fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    #plt.subplots_adjust(left=0.17, right=0.76, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.75, wspace=0.25, hspace=0.25, bottom=0.15, top=0.80)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, -0.10) )
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    save_des = f'../figure0213/n{plot_ABC}_PA_N=3.png'
    plt.savefig(save_des)
    plt.close()
    #plt.show()

def pA_pcA_p_approximation(number_opinion, p_not_A, plot_ABC):
    """plot P_A with P_{cA} for different p_{cB}

    :number_opinion: 4
    :returns: TODO

    """
    data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted_approximation.csv', header=None))
    p_cA = data[:, 0]
    p_not_cA = np.round(data[:, 1], 4)
    x_A = data[:, 2]
    x_B = data[:, 3]
    x_not_A = data[:, 4]
    #p_B_plot = np.sort(np.unique(p_cB))
    for p in p_not_A:
        index_p = np.where(p_not_cA == p)[0]
        p_cA_p = p_cA[index_p]
        p_not_cA_p = p_not_cA[index_p]
        x_A_p = x_A[index_p]
        x_B_p = x_B[index_p]
        x_not_A_p = x_not_A[index_p]
        index_sort = np.argsort(p_cA_p)
        if plot_ABC == 'A':
            y = x_A_p[index_sort] + p_cA_p[index_sort]
            ylabel = '$n_A$'
        elif plot_ABC == 'B':
            y = x_B_p[index_sort] 
            ylabel = '$n_B$'
        elif plot_ABC == 'C':
            y = x_not_A_p[index_sort] / (number_opinion-2) + p_not_cA_p[index_sort] / (number_opinion-2)
            ylabel = '$n_C$'
        elif plot_ABC == 'A_tilde':
            y = x_not_A_p[index_sort] + p_not_cA_p[index_sort] 
            ylabel = '$n_{\\tilde{A}}$'

        #y = x_B_p[index_sort]
        plt.plot(p_cA_p[index_sort], y, '-.', linewidth=lw, alpha=alpha, label='$P_{\\tilde{A}}=$' + str(p))
    plt.xlabel('$P_{A}$', fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    #plt.subplots_adjust(left=0.17, right=0.76, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)

    plt.subplots_adjust(left=0.18, right=0.75, wspace=0.25, hspace=0.25, bottom=0.15, top=0.80)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, -0.10) )
    #plt.legend(frameon=False, fontsize = legendsize, markerscale=3.0)
    plt.locator_params(nbins=5)
    save_des = f'../figure0213/n{plot_ABC}_PA_N={number_opinion}.png'
    plt.savefig(save_des)
    plt.close()
    #plt.show()

def pA_pcA_N(number_opinion_list, p):
    """plot P_A with P_{cA} for different p_{cB}

    :number_opinion: 4
    :returns: TODO

    """
    for number_opinion in number_opinion_list:
        data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted.csv', header=None))
        p_cA = data[:, 0]
        p_cB = np.round(data[:, 1], 4)
        x_A = data[:, -number_opinion]
        x_B = data[:, -number_opinion+1]
        index_p = np.where(p_cB == round(p / (number_opinion-2), 4))[0]
        p_cA_p = p_cA[index_p]
        x_A_p = x_A[index_p]
        x_B_p = x_B[index_p]
        index_sort = np.argsort(p_cA_p)
        y = x_A_p[index_sort] + p_cA_p[index_sort]
        plt.plot(p_cA_p[index_sort], y, '-.', linewidth=lw, alpha=alpha, label='$N=$' + str(number_opinion))
    plt.xlabel('$P_{cA}$', fontsize=fontsize)
    plt.ylabel('$x_A^s$', fontsize=fontsize)
    #plt.subplots_adjust(left=0.17, right=0.76, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.17, right=0.72, wspace=0.25, hspace=0.25, bottom=0.15, top=0.8)
    #plt.legend(frameon=False, fontsize = legendsize)
    plt.legend(frameon=False, fontsize = legendsize, loc='lower right', bbox_to_anchor=(1.54, -0.1))
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    plt.show()

def fluctuate_oneuncommitted(number_opinion, p, sigma, seed_list, normalization):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    for seed in seed_list:
        if normalization:
            data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_absolute/oneuncommitted_p={p}_sigma={sigma}_seed={seed}_normalization.csv', header=None))
        else:
            data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_absolute/oneuncommitted_p={p}_sigma={sigma}_seed={seed}.csv', header=None))

        p_cA = data[:, 0]
        x_A = data[:, -number_opinion]
        x_B = data[:, -number_opinion+1]
        index_sort = np.argsort(p_cA)
        y = x_A[index_sort] + p_cA[index_sort]
        plt.plot(p_cA[index_sort], y, '--', linewidth=lw, alpha=alpha, color='tab:red')
    plt.xlabel('$P_{cA}$', fontsize=fontsize)
    plt.ylabel('$x_A^s$', fontsize=fontsize)
    #plt.subplots_adjust(left=0.17, right=0.76, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.17, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    #plt.legend(frameon=False, fontsize = legendsize)
    plt.legend(frameon=False, fontsize = legendsize, loc='lower right', bbox_to_anchor=(1.54, -0.1))
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()

def tipping_point_pmax_fluctuate(number_opinion, p, sigma, seed_list, normalization, xplot):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    p_cA_tipping_list = []
    p_cA_tilde_list = []
    for seed in seed_list:
        if normalization:
            data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_absolute/oneuncommitted_p={p}_sigma={sigma}_seed={seed}_normalization.csv', header=None))
        else:
            data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_absolute/oneuncommitted_p={p}_sigma={sigma}_seed={seed}.csv', header=None))

        p_cA = data[:, 0]
        p_cAtilde = data[0, 1: number_opinion-1]
        if np.min(p_cAtilde)>=0:

            x_A = data[:, -number_opinion]
            x_B = data[:, -number_opinion+1]
            index_sort = np.argsort(p_cA)
            y = x_A[index_sort] + p_cA[index_sort]
            change = np.diff(y) 
            if np.max(change) > 0.5 * np.ptp(y):
                tipping_index = np.where(np.max(change) == change)[0][0]
                p_cA_tipping = p_cA[index_sort][tipping_index]
                if xplot == 'max':
                    if (p_cA_tipping - np.max(p_cAtilde))>=-1e-1: 
                        p_cA_tipping_list.append(p_cA_tipping)
                        p_cA_tilde_list.append(np.max(p_cAtilde))
                elif xplot == 'sd':
                    if (p_cA_tipping - np.max(p_cAtilde))>=1e-5: 
                        p_cA_tipping_list.append(p_cA_tipping)
                        p_cA_tilde_list.append(np.std(p_cAtilde))

                
    plt.plot(p_cA_tilde_list, p_cA_tipping_list, '.', linewidth=lw, alpha=alpha, label=f'$p_0={p}$')
    #plt.plot(p_cA_tilde_list, p_cA_tipping_list, '.', linewidth=lw, alpha=alpha, label=f'$\\sigma={sigma}$')
    if xplot == 'max':
        plt.xlabel('max$(P_i)$', fontsize=fontsize)
    elif xplot == 'sd':
        plt.xlabel('$SD(P_i)$', fontsize=fontsize)
    else:
        print('no available method')

    plt.ylabel('$P_{A}^{(c)}$', fontsize=fontsize)
    #plt.subplots_adjust(left=0.17, right=0.76, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.75, wspace=0.25, hspace=0.25, bottom=0.15, top=0.80)
    #plt.legend(frameon=False, fontsize = legendsize, markerscale=2.0)
    plt.legend(frameon=False, fontsize = legendsize, loc='lower right', markerscale=2, bbox_to_anchor=(1.45, -0.1))
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def tipping_point_fluctuate(number_opinion, p_list, sigma, seed_list, normalization):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    for p in p_list:
        p_cA_tipping_list = []
        p_cA_tilde_list = []
        for seed in seed_list:
            if normalization:
                data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_absolute/oneuncommitted_p={p}_sigma={sigma}_seed={seed}_normalization.csv', header=None))
            else:
                data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_absolute/oneuncommitted_p={p}_sigma={sigma}_seed={seed}.csv', header=None))

            p_cA = data[:, 0]
            p_cAtilde = data[0, 1: number_opinion-1]
            if np.min(p_cAtilde)>0:
                x_A = data[:, -number_opinion]
                x_B = data[:, -number_opinion+1]
                index_sort = np.argsort(p_cA)
                y = x_A[index_sort] + p_cA[index_sort]
                change = np.diff(y) 
                if np.max(change) > 0.5 * np.ptp(y):
                    tipping_index = np.where(np.max(change) == change)[0][0]
                    p_cA_tipping = p_cA[index_sort][tipping_index]
                    p_cA_tipping_list.append(p_cA_tipping)
                    p_cA_tilde_list.append(np.max(p_cAtilde))
        p_cA_tipping_sort = np.array(p_cA_tipping_list)[np.argsort(p_cA_tilde_list)]
        decrease_index = np.argmin(p_cA_tipping_sort)
        diff = np.abs(p_cA_tipping_sort - np.sort(p_cA_tilde_list)) 
        if np.sum(diff < 1e-4):
            decrease_index = np.where(diff< 1e-4)[0][0]
        else:
            decrease_index = -1
        
        p_cA_tipping_decrease = p_cA_tipping_sort[:decrease_index]
                
        if p!=p_list[-1]:
            plt.plot(p * np.ones(len(p_cA_tipping_decrease)), p_cA_tipping_decrease, '.', linewidth=lw, alpha=0.5, color='tab:green')
    plt.plot(p * np.ones(len(p_cA_tipping_decrease)), p_cA_tipping_decrease, '.', linewidth=lw, alpha=0.5, color='tab:green', label='$S_0$')
        #plt.plot(p * np.ones(len(p_cA_tipping_sort)), p_cA_tipping_sort, '.', linewidth=lw, alpha=alpha)
    #plt.plot(p_cA_tilde_list, p_cA_tipping_list, '.', linewidth=lw, alpha=alpha, label=f'$\\sigma={sigma}$')
    plt.xlabel('$p_0$', fontsize=fontsize)
    plt.ylabel('$P_{cA}^{(c)}$', fontsize=fontsize)
    #plt.subplots_adjust(left=0.17, right=0.76, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=2.0)
    #plt.legend(frameon=False, fontsize = legendsize, loc='lower right', bbox_to_anchor=(1.54, -0.1))
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def tipping_point(number_opinion, approximation):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    p_cA_tipping_list = []
    p_not_cA_list = []
    if approximation:
        data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted_approximation.csv', header=None))
        p_cA = data[:, 0]
        p_not_cA = np.round(data[:, 1], 4)
        p_not_cA_unique = np.sort(np.unique(p_not_cA))
        x_A = data[:, 2]
        x_B = data[:, 3]
        x_not_A = data[:, 4]

    else:
        data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted.csv', header=None))
        p_cA = data[:, 0]
        p_not_cA = data[:, 1]
        p_not_cA_unique = np.sort(np.unique(p_not_cA))
        x_A = data[:, -number_opinion]
        x_B = data[:, -number_opinion+1]

    for p in p_not_cA_unique:
        index_p = np.where(p_not_cA == p)[0]
        p_cA_p = p_cA[index_p]
        x_A_p = x_A[index_p]
        x_B_p = x_B[index_p]
        index_sort = np.argsort(p_cA_p)
        y = x_A_p[index_sort] + p_cA_p[index_sort]
        change = y[2:] - y[:-2]
        if np.max(change) > 0.5* np.ptp(y):
            tipping_index = np.where(np.max(change) == change)[0][0]
            p_cA_tipping = p_cA_p[index_sort][tipping_index+1]
            p_not_cA_list.append(p)
            p_cA_tipping_list.append(p_cA_tipping)
    plt.plot(np.array(p_not_cA_list), p_cA_tipping_list, '-', linewidth=lw, alpha=alpha, label=f'm={number_opinion}')
    #plt.plot(np.array(p_not_cA_list)/(number_opinion-2), p_cA_tipping_list, '-', linewidth=lw, alpha=alpha, label='$S_1$', color='tab:red')
    plt.xlabel('$p_0$', fontsize=fontsize)
    plt.xlabel('$P_{\\tilde{A}}$', fontsize=fontsize)
    plt.ylabel('$P_{A}^{(c)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.locator_params(nbins=5)

    plt.subplots_adjust(left=0.18, right=0.75, wspace=0.25, hspace=0.25, bottom=0.15, top=0.80)
    #plt.legend(frameon=False, fontsize = legendsize, markerscale=3.0)
    plt.legend(frameon=False, fontsize = legendsize, loc='lower right', bbox_to_anchor=(1.45, 0.1))
    #plt.show()
    return None

def triple_point(number_opinion, approximation):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    p_cA_tipping_A = []
    p_cA_tipping_B = []
    p_not_cA_A = []
    p_not_cA_B = []
    if approximation:
        data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted_approximation.csv', header=None))
        p_cA = data[:, 0]
        p_not_cA = np.round(data[:, 1], 4)
        p_not_cA_unique = np.sort(np.unique(p_not_cA))
        x_A = data[:, 2]
        x_B = data[:, 3]
        x_not_A = data[:, 4]

    else:
        data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted.csv', header=None))
        p_cA = data[:, 0]
        p_not_cA = data[:, 1]
        p_not_cA_unique = np.sort(np.unique(p_not_cA))
        x_A = data[:, -number_opinion]
        x_B = data[:, -number_opinion+1]

    for p in p_not_cA_unique:
        index_p = np.where(p_not_cA == p)[0]
        p_cA_p = p_cA[index_p]
        x_A_p = x_A[index_p]
        x_B_p = x_B[index_p]
        index_sort = np.argsort(p_cA_p)
        y_A = x_A_p[index_sort] + p_cA_p[index_sort]
        y_B = x_B_p[index_sort]
        change_A = y_A[2:] - y_A[:-2]
        change_B = np.abs(y_B[2:] - y_B[:-2])
        if np.max(change_A) > 0.5 * np.ptp(y_A):
            tipping_A = np.where(np.max(change_A) == change_A)[0][0]
            p_cA_tipping = p_cA_p[index_sort][tipping_A+1]
            p_not_cA_A.append(p)
            p_cA_tipping_A.append(p_cA_tipping)
        if np.max(change_B) > 0.1 * np.ptp(y_B) and np.max(y_B) > 1/3 *(1-p-np.max(p_cA_p)):
            tipping_B = np.where(np.max(change_B) == change_B)[0][0]
            p_cA_tipping = p_cA_p[index_sort][tipping_B+1]
            p_not_cA_B.append(p)
            p_cA_tipping_B.append(p_cA_tipping)
    plt.plot(np.array(p_not_cA_A)/(number_opinion-2), p_cA_tipping_A, '-', linewidth=lw, alpha=alpha*0.7, label=f'N={number_opinion}', color='k')
    plt.plot(np.array(p_not_cA_B)/(number_opinion-2), p_cA_tipping_B, '.', linewidth=lw, alpha=alpha*0.7, label=f'N={number_opinion}', color='tab:red')
    plt.xlabel('$P_0$', fontsize=fontsize)
    plt.ylabel('$P_{cA}^{(c)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    #plt.legend(frameon=False, fontsize = legendsize, markerscale=2.0)
    #plt.legend(frameon=False, fontsize = legendsize, loc='lower right', bbox_to_anchor=(1.54, -0.1))
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def tipping_point_two(number_opinion, approximation):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    p_cA_tipping_list = []
    p_not_cA_list = []
    if approximation:
        data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_two.csv', header=None))
        p_not_cA = np.round(data[:, 0], 4)
        p_not_cA_unique = np.sort(np.unique(p_not_cA))
        x_A = data[:, 1]
        x_not_A = data[:, 2]

    else:
        data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted_two.csv', header=None))
        p_cA = data[:, 0]
        p_not_cA = data[:, 1]
        p_not_cA_unique = np.sort(np.unique(p_not_cA))
        x_A = data[:, -number_opinion]
        x_B = data[:, -number_opinion+1]

    index_sort = np.argsort(p_not_cA)
    x_A_sort = x_A[index_sort]
    x_not_A_sort = x_not_A[index_sort]
    y = x_not_A_sort + np.sort(p_not_cA)
    y = x_A_sort
    plt.plot(np.sort(p_not_cA)/(number_opinion-1) *(number_opinion-2), y, '-', linewidth=lw, alpha=alpha, label=f'N={number_opinion}')
    plt.xlabel('$P_{c\\tilde{A}}$', fontsize=fontsize)
    plt.ylabel('$x_B$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=2.0)
    #plt.legend(frameon=False, fontsize = legendsize, loc='lower right', bbox_to_anchor=(1.54, -0.1))
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def turning_point_N(number_opinion_list, approximation):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    p_cA_turning_list = []
    p_not_cA_list = []
    for number_opinion in number_opinion_list:
        if approximation:
            data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_two.csv', header=None))
            p_not_cA = np.round(data[:, 0], 4)
            p_not_cA_unique = np.sort(np.unique(p_not_cA))
            x_A = data[:, 1]
            x_not_A = data[:, 2]

        else:
            data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted_two.csv', header=None))
            p_cA = data[:, 0]
            p_not_cA = data[:, 1]
            p_not_cA_unique = np.sort(np.unique(p_not_cA))
            x_A = data[:, -number_opinion]
            x_B = data[:, -number_opinion+1]

        index_sort = np.argsort(p_not_cA)
        x_A_sort = x_A[index_sort]
        x_not_A_sort = x_not_A[index_sort]
        y = x_not_A_sort + np.sort(p_not_cA)
        y = x_A_sort
        change = np.abs(y[2:] - y[:-2])
        if np.max(change) > 0.1 * np.ptp(y):
            turning_index = np.where(np.max(change) == change)[0][0]
            p_turning = p_not_cA[index_sort][turning_index+1]

            p_cA_turning_list.append(p_turning)
    number_opinion_list = np.array(number_opinion_list)
    plt.plot(number_opinion_list, np.array(p_cA_turning_list)/(number_opinion_list-1) , '-', marker='o', linewidth=lw, alpha=alpha)
    plt.xlabel('$N$', fontsize=fontsize)
    plt.ylabel('$p_{0}}^{(c)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=2.0)
    #plt.legend(frameon=False, fontsize = legendsize, loc='lower right', bbox_to_anchor=(1.54, -0.1))
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def tipping_pmax(number_opinion, p_list):
    """TODO: Docstring for polarization.

    :arg1: TODO
    :returns: TODO

    """

    for p in p_list:
        des = f'../data/num_opinion={number_opinion}_lowerbound/'
        des_file = des  + f'oneuncommitted_p={p}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        p_cAtilde = p * (number_opinion-2)
        p_cC_list = data[:, 0]
        p_cCtilde_list = np.round(data[:, 1], 14)
        xC_list = data[:, 2]
        xB_list = data[:, 3]
        xCtilde_list = data[:, 4]
        pmax_n = p_cAtilde - p_cC_list
        pmax_list = p_cCtilde_list - pmax_n
        p_cCtilde_unique = np.unique(p_cCtilde_list)
        index_sort = np.argsort(pmax_list)
        y = xB_list[index_sort]
        y = xCtilde_list[index_sort] + p_cCtilde_list[index_sort]
        #y = xC_list[index_sort] + p_cC_list[index_sort]
        #plt.plot(pmax_list[index_sort], y)
        plt.plot(pmax_list, xB_list, '.')

def fluctuate_lowerbound(number_opinion, p_list):
    """TODO: Docstring for polarization.

    :arg1: TODO
    :returns: TODO

    """
    p_cA_turning_list = []
    for p in p_list:
        des = f'../data/num_opinion={number_opinion}_lowerbound/'
        des_file = des  + f'oneuncommitted_p={p}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        p_cAtilde = p * (number_opinion-2)
        p_cC_list = data[:, 0]
        p_cCtilde_list = np.round(data[:, 1], 14)
        xC_list = data[:, 2]
        xB_list = data[:, 3]
        xCtilde_list = data[:, 4]
        pmax_n = p_cAtilde - p_cC_list
        pmax_list = p_cCtilde_list - pmax_n
        p_cCtilde_unique = np.unique(p_cCtilde_list)
        index_sort = np.argsort(pmax_list)
        y = xCtilde_list[index_sort] + p_cCtilde_list[index_sort]
        y = xB_list[index_sort]
        change = np.abs(y[8:] - y[6:-2])
        if np.max(change) > 0.5 * np.ptp(y) and np.max(change)>0.01:
            tipping_index = np.where(np.max(change) == change)[0][0]
            p_cA_turning = pmax_list[index_sort][tipping_index+1]
            p_cA_turning_list.append(p_cA_turning)
            #p_cA_turning_list.append(0)
        else:
            des = f'../data/lowerbound2/'
            des_file = des  + f'oneuncommitted_p={round(p*(number_opinion-2), 3)}.csv'
            data = np.array(pd.read_csv(des_file, header=None))
            p_cA = data[:, 0]
            xA = data[:, 6]
            index_sort= np.argsort(p_cA)
            y = xA[index_sort] + p_cA[index_sort]
            change = np.abs(y[2:] - y[:-2])
            if np.max(change) > 0.5 * np.ptp(y):
                tipping_index = np.where(np.max(change) == change)[0][0]
                p_cA_turning_list.append(np.sort(p_cA)[tipping_index])
                #p_cA_turning_list.append(0)
            else:
                print(p, np.max(change))
    plt.plot(p_list, p_cA_turning_list, color='tab:blue', alpha=alpha, linewidth=lw, label='$S_2$')
    return None

def original_lowerbound(number_opinion, p):
    """TODO: Docstring for original_lowerbound.

    :number_opinion: TODO
    :p: TODO
    :returns: TODO

    """
    des = f'../data/num_opinion={number_opinion}_original_lowerbound/'
    des_file = des + f'p={p}_zoomin2.csv'
    des_file = des + f'p={p}.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    p_cA_list = data[:, 0]
    p_cC_list= np.round(data[:, 2], 10)
    p_cD_list = np.round(data[:, 3], 10)
    p_cE_list = np.round(data[:, 4], 10)
    x_A_list = data[:, -5]
    x_B_list = data[:, -4]
    x_C_list = data[:, -3]
    p_cCD_unique = np.unique(np.vstack((p_cC_list, p_cD_list)), axis=1).transpose()
    p_cCDE_unique = np.unique(np.vstack((p_cC_list, p_cD_list, p_cE_list)), axis=1).transpose()
    p_cA_tipping_list = np.zeros((len(p_cCD_unique)))
    index_list = []
    for p_cCD, i in zip(p_cCD_unique, range(len(p_cCD_unique))):
        p_cC, p_cD = p_cCD
        index = np.where((p_cC == p_cC_list) & (p_cD == p_cD_list))[0]
        p_cA = p_cA_list[index]
        x_A = x_A_list[index]
        x_B = x_B_list[index]
        x_C = x_C_list[index]
        index_sort = np.argsort(p_cA)
        x_A_sort = x_A[index_sort]
        x_B_sort = x_B[index_sort]
        x_C_sort = x_C[index_sort]
        p_cA_sort =p_cA[index_sort]
        y = x_B_sort
        y = x_A_sort + p_cA_sort 
        change = y[1:] - y[:-1]
        if np.max(change) > 0.5 * np.ptp(y):
            tipping_index = np.where(np.max(change) == change)[0][0]
            p_cA_tipping = p_cA_sort[tipping_index+1]
            if np.abs(p_cA_tipping - p_cC) > 1e-5:
                p_cA_tipping_list[i] = p_cA_tipping
                index_list.append(i)
    index_list = np.array(index_list)
    #x = np.linspace(np.min(p_cA_tipping_list[index_list]), np.max(p_cA_tipping_list[index_list]), 5)
    
    plt.plot(p_cCDE_unique[index_list, 0], p_cA_tipping_list[index_list], 'o', markerfacecolor="None", markeredgecolor='tab:blue', label='C')
    plt.plot(p_cCDE_unique[index_list, 1], p_cA_tipping_list[index_list], '*', markerfacecolor="None", markeredgecolor='tab:red', label='D')
    plt.plot(p_cCDE_unique[index_list, 2], p_cA_tipping_list[index_list], '^', markerfacecolor="None", markeredgecolor='tab:green', label='E')
    #plt.plot(x, x, 'tab:grey')
    plt.xlabel('$P_{ci}$', fontsize=fontsize)
    plt.ylabel('$P_{cA}^{(c)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.24, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=2.0)
    plt.locator_params(nbins=5)
    #plt.show()
    return p_cCD_unique, p_cA_tipping_list 

def pcA_psigma(number_opinion, p):
    """TODO: Docstring for pcA_psigma.

    :number_opinion: TODO
    :p: TODO
    :returns: TODO

    """
    des = f'../data/num_opinion={number_opinion}_original_lowerbound/'
    des_file = des + f'p={p}_zoomin.csv'
    des_file = des + f'p={p}.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    p_cA_list = data[:, 0]
    p_cC_list= np.round(data[:, 2], 10)
    p_cD_list = np.round(data[:, 3], 10)
    p_cE_list = np.round(data[:, 4], 10)
    x_A_list = data[:, -5]
    x_B_list = data[:, -4]
    x_C_list = data[:, -3]
    p_cCD_unique = np.unique(np.vstack((p_cC_list, p_cD_list)), axis=1).transpose()
    p_cCDE_unique = np.unique(np.vstack((p_cC_list, p_cD_list, p_cE_list)), axis=1).transpose()
    p_std_list = np.std(p_cCDE_unique, axis=1)
    p_cA_tipping_list = np.zeros((len(p_cCD_unique)))
    index_list = []
    for p_cCD, i in zip(p_cCD_unique, range(len(p_cCD_unique))):
        p_cC, p_cD = p_cCD
        index = np.where((p_cC == p_cC_list) & (p_cD == p_cD_list))[0]
        p_cA = p_cA_list[index]
        x_A = x_A_list[index]
        x_B = x_B_list[index]
        x_C = x_C_list[index]
        index_sort = np.argsort(p_cA)
        x_A_sort = x_A[index_sort]
        x_B_sort = x_B[index_sort]
        x_C_sort = x_C[index_sort]
        p_cA_sort =p_cA[index_sort]
        y = x_A_sort + p_cA_sort 
        change = y[1:] - y[:-1]
        if np.max(change) > 0.5 * np.ptp(y):
            tipping_index = np.where(np.max(change) == change)[0][0]
            p_cA_tipping = p_cA_sort[tipping_index+1]
            if np.abs(p_cA_tipping - p_cC) > 1e-4:
                p_cA_tipping_list[i] = p_cA_tipping
                index_list.append(i)
    index_list = np.array(index_list)
    #plt.plot(p_std_list[index_list], p_cA_tipping_list[index_list], '.')
    plt.plot(p_std_list[index_list], p_cA_tipping_list[index_list], '.', label=f'$p_0={p}$')
    plt.xlabel('$SD(P_{ci})$', fontsize=fontsize)
    plt.ylabel('$P_{cA}^{(c)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=2.0)
    #plt.legend(frameon=False, fontsize = legendsize, loc='lower right', bbox_to_anchor=(1.54, -0.1))
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()

    return None

def xA_pA_bounds(number_opinion, p):
    """TODO: Docstring for xA_pA_bounds.

    :number_opinion: TODO
    :p: TODO
    :returns: TODO

    """
    des_four = f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_four/'
    des_three = f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_three/'
    des_file_three = des_three + f'p0={p}.csv'
    des_file_four = des_four + f'p0={p}.csv'
    data_three = np.array(pd.read_csv(des_file_three, header=None))
    data_four = np.array(pd.read_csv(des_file_four, header=None))
    "four"
    p_A = data_four[:, 0]
    x_A = data_four[:, 3]
    y_A = x_A + p_A
    index_sort = np.argsort(p_A)
    p_A_sort = p_A[index_sort]
    y_A_sort = y_A[index_sort]
    plt.plot(p_A_sort, y_A_sort)
    "three"
    p_A = data_three[:, 0]
    x_A = data_three[:, 2]
    y_A = x_A + p_A
    index_sort = np.argsort(p_A)
    p_A_sort = p_A[index_sort]
    y_A_sort = y_A[index_sort]
    plt.plot(p_A_sort, y_A_sort)

    plt.show()
    return None

def xA_pA_bounds_original(number_opinion, p):
    """TODO: Docstring for xA_pA_bounds.

    :number_opinion: TODO
    :p: TODO
    :returns: TODO

    """
    des_four = f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_four/'
    des_three = f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_three/'
    des_original = f'../data/num_opinion={number_opinion}_original/'
    des_file_three = des_three + f'p0={p}.csv'
    des_file_four = des_four + f'p0={p}.csv'
    des_file_original = des_original + f'p={p}_random.csv'
    data_three = np.array(pd.read_csv(des_file_three, header=None))
    data_four = np.array(pd.read_csv(des_file_four, header=None))
    data_original = np.array(pd.read_csv(des_file_original, header=None))
    "three"
    p_A = data_three[:, 0]
    x_A = data_three[:, 2]
    y_A = x_A + p_A
    index_sort = np.argsort(p_A)
    p_A_sort = p_A[index_sort]
    y_A_sort = y_A[index_sort]
    plt.plot(p_A_sort, y_A_sort, color='tab:red', linewidth=lw, alpha=alpha, label='$S_1$')
    "four"
    p_A = data_four[:, 0]
    x_A = data_four[:, 3]
    y_A = x_A + p_A
    index_sort = np.argsort(p_A)
    p_A_sort = p_A[index_sort]
    y_A_sort = y_A[index_sort]
    plt.plot(p_A_sort, y_A_sort, color='tab:blue', linewidth=lw, alpha=alpha, label='$S_2$')
    "original"
    p_A = data_original[:, 0]
    x_A = data_original[:, -number_opinion]
    y_A = x_A + p_A
    index_sort = np.argsort(p_A)
    p_A_sort = p_A[index_sort]
    y_A_sort = y_A[index_sort]
    plt.plot(p_A_sort, y_A_sort, '.', color='tab:green', alpha=0.5, markersize=5, label='$S_0$')
    plt.xlabel('$P_{A}$', fontsize=fontsize)
    plt.ylabel('$n_A$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=3.0)
    plt.locator_params(nbins=6)
    save_des = f'../manuscript/0329/figure/nA_number_opinion={number_opinion}_p={p}_random.png'
    plt.savefig(save_des)
    plt.close('all')
    #plt.show()
    return None

def xs_pA_3D(number_opinion, p_list):
    """TODO: Docstring for xs_pA_3D.

    :number_opinion: TODO
    :p_list: TODO
    :returns: TODO

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for p in p_list:
        des_four = f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_four/'
        des_three = f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_three/'
        des_file_three = des_three + f'p0={p}.csv'
        des_file_four = des_four + f'p0={p}.csv'
        data_three = np.array(pd.read_csv(des_file_three, header=None))
        data_four = np.array(pd.read_csv(des_file_four, header=None))
        "four"
        p_A = data_four[:, 0]
        x_A = data_four[:, 3]
        y_A = x_A + p_A
        index_sort = np.argsort(p_A)
        p_A_sort = p_A[index_sort]
        y_A_sort = y_A[index_sort]
        ax.scatter3D(p_A_sort, p * np.ones(len(p_A_sort)), y_A_sort, color='tab:blue')
        "three"
        p_A = data_three[:, 0]
        x_A = data_three[:, 2]
        y_A = x_A + p_A
        index_sort = np.argsort(p_A)
        p_A_sort = p_A[index_sort]
        y_A_sort = y_A[index_sort]
        ax.scatter3D(p_A_sort, p * np.ones(len(p_A_sort)), y_A_sort, color='tab:red')
    return None

def xA_pA_bounds_gamma_original(number_opinion, gamma):
    """TODO: Docstring for xA_pA_bounds.

    :number_opinion: TODO
    :p: TODO
    :returns: TODO

    """
    des_four = f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_four/'
    des_three = f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_three/'
    des_original = f'../data/num_opinion={number_opinion}_original/'
    des_file_three = des_three + f'gamma={gamma}.csv'
    des_file_four = des_four + f'gamma={gamma}.csv'
    des_file_original = des_original + f'gamma={gamma}_random.csv'
    data_three = np.array(pd.read_csv(des_file_three, header=None))
    data_four = np.array(pd.read_csv(des_file_four, header=None))
    data_original = np.array(pd.read_csv(des_file_original, header=None))
    "four"
    p_A = data_four[:, 0]
    x_A = data_four[:, 3]
    y_A = x_A + p_A
    index_sort = np.argsort(p_A)
    p_A_sort = p_A[index_sort]
    y_A_sort = y_A[index_sort]
    plt.plot(p_A_sort, y_A_sort, color='tab:blue', linewidth=lw, alpha=alpha, label='$S_2$')
    "three"
    p_A = data_three[:, 0]
    x_A = data_three[:, 2]
    y_A = x_A + p_A
    index_sort = np.argsort(p_A)
    p_A_sort = p_A[index_sort]
    y_A_sort = y_A[index_sort]
    plt.plot(p_A_sort, y_A_sort, color='tab:red', linewidth=lw, alpha=alpha, label='$S_1$')
    "original"
    p_A = data_original[:, 0]
    x_A = data_original[:, -number_opinion]
    y_A = x_A + p_A
    index_sort = np.argsort(p_A)
    p_A_sort = p_A[index_sort]
    y_A_sort = y_A[index_sort]
    plt.plot(p_A_sort, y_A_sort, '.', color='tab:green', alpha=0.5, markersize=5, label='original')
    plt.xlabel('$P_{cA}$', fontsize=fontsize)
    plt.ylabel('$x_A^{(s)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=3.0)
    plt.locator_params(nbins=6)
    save_des = f'../report/report031421/xA_number_opinion={number_opinion}_gamma={gamma}_random.png'
    plt.savefig(save_des)
    plt.close('all')
    #plt.show()
    return None

def xA_N_bounds(N_list, pA, pAtilde):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    des_three = f'../data/Nchange_oneuncommitted_approximation_three/'
    des_four = f'../data/Nchange_oneuncommitted_approximation_four/'
    des_file_four = des_four + f'pA={pA}_pAtilde={pAtilde}.csv'
    "three"
    xA = []
    xB = []
    xAtilde = []
    for number_opinion in N_list:
        des_file_three = des_three + f'num_opinion={number_opinion}.csv'
        data_three = np.array(pd.read_csv(des_file_three, header=None))
        index = np.where((np.abs(data_three[:, 0] - pA)<1e-10) & (np.abs(data_three[:, 1] - pAtilde)<1e-10))[0]
        xA.append(data_three[index, 2])
        xB.append(data_three[index, 3])
        xAtilde.append(data_three[index, 4])
    p0 = pAtilde/(N_list - 2)
    index = np.where(pA > p0)[0]
    plt.plot(N_list[index], np.array(xA)[index] + pA, '-o', markersize=3, linewidth=lw, alpha=alpha, label='$p_{\\tilde{A}}=$' + f'{pAtilde}')
    #plt.plot(N_list, xB)
    #plt.plot(N_list, xAtilde + pAtilde)
    "four"
    data_four = np.array(pd.read_csv(des_file_four, header=None))
    xA = data_four[:, 3]
    xB = data_four[:, 4]
    xC = data_four[:, 5]
    xCtilde = data_four[:, 6]
    #plt.plot(N_list[index], xA[index] + pA, '-o', linewidth=lw, alpha=alpha, label='$p_{c\\tilde{A}}=$' + f'{pAtilde}')
    #plt.plot(N_list[index], np.ones(len(index)) *(xA[-1] + pA), '-o', linewidth=lw, alpha=alpha, label='$p_{c\\tilde{A}}=$' + f'{pAtilde}')
    #plt.plot(N_list, xB)
    #plt.plot(N_list, xAtilde + pAtilde)
    plt.xlabel('$N$', fontsize=fontsize)
    plt.ylabel('$x_A^{(s)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.75, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, 0.))
    plt.locator_params(nbins=6)

def xA_N_approximation_four(pA_list, pAtilde_list):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    des_four = f'../data/Nchange_oneuncommitted_approximation_four/'
    N_list = np.arange(3, 10, 1)
    xA_list = np.zeros((len(pA_list), len(pAtilde_list)))
    for pA, i in zip(pA_list, range(len(pA_list))):
        for pAtilde, j in zip(pAtilde_list, range(len(pAtilde_list))):
            des_file_four = des_four + f'pA={pA}_pAtilde={pAtilde}.csv'
            data_four = np.array(pd.read_csv(des_file_four, header=None))
            xA = data_four[0, 3]
            xB = data_four[0, 4]
            xC = data_four[0, 5]
            xCtilde = data_four[0, 6]
            xA_list[i, j] = xA + pA
    for pAtilde, j in zip(pAtilde_list, range(len(pAtilde_list))):
        plt.plot(pA_list, xA_list[:, j], '-o', linewidth=lw, alpha=alpha, label='$p_{c\\tilde{A}}=$' + f'{pAtilde}')
    plt.xlabel('$P_{cA}$', fontsize=fontsize)
    plt.ylabel('$x_A^{(s)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.75, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, 0.))
    plt.locator_params(nbins=6)

def xA_N_bounds_original(pA, pAtilde):
    """TODO: Docstring for xA_N.

    :pA: TODO
    :pAtilde: TODO
    :returns: TODO

    """
    des_three = f'../data/Nchange_oneuncommitted_approximation_three/'
    des_four = f'../data/Nchange_oneuncommitted_approximation_four/'
    des_original = f'../data/Nchange_original/'
    des_file_three = des_three + f'pA={pA}_pAtilde={pAtilde}.csv'
    des_file_four = des_four + f'pA={pA}_pAtilde={pAtilde}.csv'
    data_three = np.array(pd.read_csv(des_file_three, header=None))
    data_four = np.array(pd.read_csv(des_file_four, header=None))
    "three"
    xA = data_three[:, 2]
    xB = data_three[:, 3]
    xAtilde = data_three[:, 4]
    N_list = np.arange(3, 9, 1)
    p0 = pAtilde/(N_list - 2)
    index = np.where(pA > p0)[0]
    N_index = N_list[index]
    plt.plot(N_index, xA[index] + pA, '-o', linewidth=lw, markersize=4, alpha=alpha, label='$S_1$')
    #plt.plot(N_list, xB)
    #plt.plot(N_list, xAtilde + pAtilde)
    "four"
    xA = data_four[:, 3]
    xB = data_four[:, 4]
    xC = data_four[:, 5]
    xCtilde = data_four[:, 6]
    plt.plot(N_index, xA[index] + pA, '-o', linewidth=lw, markersize=4, alpha=alpha, label='$S_2$')
    #plt.plot(N_list, xB)
    #plt.plot(N_list, xAtilde + pAtilde)

    "original"
    for N in N_index:
        des_original_N = des_original + f'num_opinion={N}/'
        des_file_original = des_original_N + f'pA={pA}_pAtilde={pAtilde}_random.csv'
        data_original = np.array(pd.read_csv(des_file_original, header=None))
        xA = data_original[:, -N]
        if N == N_index[0]:
            plt.plot(np.ones(len(xA)) * N, xA + pA, '*', linewidth=lw, markersize=8, alpha=alpha, color='tab:green', label='original')
        else:
            plt.plot(np.ones(len(xA)) * N, xA + pA, '*', linewidth=lw, markersize=8, alpha=alpha, color='tab:green')

    plt.xlabel('$N$', fontsize=fontsize)
    plt.ylabel('$x_A^{(s)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.98, wspace=0.25, hspace=0.25, bottom=0.15, top=0.98)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0)
    plt.locator_params(nbins=6)
    
def delta_x_delta_p(p_sum_list):
    """TODO: Docstring for delta_x_delta_p.

    :p_sum: TODO
    :returns: TODO

    """
    number_opinion = 3
    des = f'../data/num_opinion={number_opinion}_dx_dp/'
    for p_sum in p_sum_list:
        des_file = des + f'p_sum={p_sum}.csv'
        data = np.array(pd.read_csv(des_file, header=None))
        pA = data[:, 0]
        pC = data[:, 2]
        xA = data[:, -3]
        xC = data[:, -1]
        delta_p = pA-pC
        delta_x = xA-xC 
        index_sort = np.argsort(delta_p)
        delta_p_sort = delta_p[index_sort] 
        #delta_x_sort = delta_x[index_sort]+ delta_p_sort
        delta_x_sort = delta_x[index_sort]
        delta_x_sort[0] = delta_p_sort[0]
        x = delta_p_sort
        x = delta_p_sort/p_sum
        plt.plot(x, delta_x_sort, 'o-', linewidth=lw, markersize=3, alpha=alpha, label=f'$P_c=${p_sum}')
    plt.xlabel('$\\Delta P/P_c$', fontsize=fontsize)
    plt.ylabel('$\\Delta x$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.75, wspace=0.25, hspace=0.25, bottom=0.15, top=0.8)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, 0.) )
    plt.locator_params(nbins=6)
    #plt.show()
    return None
    
def xA_pA_two_opinion(one_or_two_committed):
    """TODO: Docstring for xA_pA_bounds.

    :number_opinion: TODO
    :p: TODO
    :returns: TODO

    """
    number_opinion = 2
    des = f'../data/num_opinion={number_opinion}_original/'
    if one_or_two_committed == 1:
        des_file = des + f'oneuncommitted.csv'
    else:
        des_file = des + f'nouncommitted.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    p_A = data[:, 0]
    x_A = data[:, -2]
    p_B = data[:, 1]
    x_B = data[:, -1]
    y_A = x_A + p_A
    y_B = x_B + p_B
    index_sort = np.argsort(p_A)
    p_A_sort = p_A[index_sort]
    p_B_sort = p_B[index_sort]
    y_A_sort = y_A[index_sort]
    y_B_sort = y_B[index_sort]
    if one_or_two_committed == 1:
        plt.plot(p_A_sort, y_A_sort, linewidth = lw, alpha=alpha)
    else:
        for pB in np.unique(np.round(p_B_sort, 10))[::60]:
            index = np.where(abs(p_B_sort - pB) < 1e-10)[0]
            #plt.plot(p_A_sort[index], y_A_sort[index], linewidth = lw, alpha=alpha, label=f'$P_B={pB}$')
    plt.xlabel('$P_B$', fontsize=fontsize)
    plt.ylabel('$P_A^{(c)}$', fontsize=fontsize)

    pB_discontinuous = []
    pB_continuous = []
    pA_discontinuous = []
    pA_continuous = []
    for pB in np.unique(np.round(p_B_sort, 10)):
        pB_index = np.where(abs(p_B_sort - pB) < 1e-10)[0]
        index_dominance = np.where(y_A_sort[pB_index] > y_B_sort[pB_index])[0]
        if len(index_dominance):
            pA = p_A_sort[pB_index][index_dominance[0]]
            if np.max(np.diff(y_A_sort[pB_index])) > 0.05:
                pB_discontinuous.append(pB) 
                pA_discontinuous.append(pA) 
            else:
                pB_continuous.append(pB) 
                pA_continuous.append(pA) 

    plt.plot(pB_discontinuous, pA_discontinuous, '.', label='discontinuous')
    plt.plot(pB_continuous, pA_continuous, '.', label='continuous')
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.75, wspace=0.25, hspace=0.25, bottom=0.15, top=0.80)
    #plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, -0.10) )
    plt.legend(frameon=False, fontsize = legendsize, markerscale=3.0)
    plt.locator_params(nbins=6)
    
    return None

def tipping_point_continuous(number_opinion, approximation):
    """TODO: Docstring for fluctuate_oneuncommitted.

    :number_opinion: TODO
    :p: TODO
    :sigma: TODO
    :returns: TODO

    """
    p_cA_tipping_list = []
    p_not_cA_list = []
    p_cA_continuous_list = []
    p_not_cA_continuous_list = []
    if approximation:
        data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted_approximation.csv', header=None))
        p_cA = data[:, 0]
        p_not_cA = np.round(data[:, 1], 4)
        p_not_cA_unique = np.sort(np.unique(p_not_cA))
        x_A = data[:, 2]
        x_B = data[:, 3]
        x_not_A = data[:, 4]

    else:
        data = np.array(pd.read_csv(f'../data/num_opinion={number_opinion}_oneuncommitted.csv', header=None))
        p_cA = data[:, 0]
        p_not_cA = data[:, 1]
        p_not_cA_unique = np.sort(np.unique(p_not_cA))
        x_A = data[:, -number_opinion]
        x_B = data[:, -number_opinion+1]

    for p in p_not_cA_unique:
        index_p = np.where(p_not_cA == p)[0]
        p_cA_p = p_cA[index_p]
        x_A_p = x_A[index_p]
        x_B_p = x_B[index_p]
        x_not_A_p = x_not_A[index_p]
        index_sort = np.argsort(p_cA_p)
        y = x_A_p[index_sort] + p_cA_p[index_sort]
        change = y[2:] - y[:-2]
        if np.max(change) > 0.06* np.ptp(y):
            tipping_index = np.where(np.max(change) == change)[0][0]
            p_cA_tipping = p_cA_p[index_sort][tipping_index+1]
            p_not_cA_list.append(p)
            p_cA_tipping_list.append(p_cA_tipping)
        else:
            tipping_index = np.where(y > x_not_A_p[index_sort] + p )[0][0]
            p_cA_continuous = p_cA_p[index_sort][tipping_index]
            p_not_cA_continuous_list.append(p)
            p_cA_continuous_list.append(p_cA_continuous)
    plt.plot(np.array(p_not_cA_list)/(number_opinion-2), p_cA_tipping_list, '.', linewidth=lw, alpha=alpha, label='discontinuous', color='tab:blue')
    plt.plot(np.array(p_not_cA_continuous_list)/(number_opinion-2), p_cA_continuous_list, '.', linewidth=lw, alpha=alpha, label='continuous', color='tab:red')
    plt.xlabel('$P_C$', fontsize=fontsize)
    plt.ylabel('$P_A^{(c)}$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.75, wspace=0.25, hspace=0.25, bottom=0.15, top=0.80)
    #plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, -0.10) )
    plt.legend(frameon=False, fontsize = legendsize, markerscale=3.0)
    #plt.legend(frameon=False, fontsize = legendsize, loc='lower right', bbox_to_anchor=(1.54, -0.1))
    #legend = plt.legend(frameon=False, fontsize = legendsize, loc='upper right', markerscale=1.5, bbox_to_anchor=(1.45, 1.))
    plt.locator_params(nbins=5)
    #plt.show()
    return None

def peak_A(number_opinion):
    """TODO: Docstring for delta_x_delta_p.

    :p_sum: TODO
    :returns: TODO

    """
    des = f'../data/num_peak/'
    des_file = des + f'number_opinion={number_opinion}.csv'
    data = np.array(pd.read_csv(des_file, header=None))
    pA_list = np.round(data[:, 0], 10)
    gamma_list = np.round(data[:, 1], 10)
    peaks_list = data[:, 2]
    pA_unique = np.unique(pA_list)
    gamma_unique = np.unique(gamma_list)
    pA_plot = []
    gamma_critical1_plot = []
    for pA in pA_unique:
        index = np.where(pA_list == pA)[0]
        gamma_pA = gamma_list[index]
        peaks_pA = peaks_list[index]
        gamma_sort = np.sort(gamma_pA)
        gamma_argsort = np.argsort(gamma_pA)
        peaks_sort = peaks_pA[gamma_argsort]
        if np.max(peaks_sort[0]) == 2 and peaks_sort[-1] == 1:
            peak_one = np.where(peaks_sort == 1)[0]
            peak_two = np.where(peaks_sort == 2)[0]
            print(np.max(peaks_sort[0]))
            if not np.any(np.diff(peak_one)) > 1:
                if len(peak_two) == 1:
                    gamma_critical1 = gamma_sort[peak_two[-1]+1]
                else:
                    gamma_critical1 = gamma_sort[peak_two[-2]+2]
                gamma_critical2 = gamma_sort[peak_one[0]]
                pA_plot.append(pA)
                gamma_critical1_plot.append(gamma_critical1)
            else:
                print('wrong')
    plt.plot(pA_plot, gamma_critical1_plot, 'o-',  linewidth=lw, markersize=3, alpha=alpha, label=f'N={number_opinion}')
    #plt.plot(pA, gamma_critical2, 'o-', color='tab:blue', linewidth=lw, markersize=3, alpha=alpha)

    plt.xlabel('$P_A$', fontsize=fontsize)
    plt.ylabel('$\\gamma$', fontsize=fontsize)
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    plt.subplots_adjust(left=0.18, right=0.75, wspace=0.25, hspace=0.25, bottom=0.15, top=0.8)
    plt.legend(frameon=False, fontsize = legendsize, markerscale=1.0, loc='lower right', bbox_to_anchor=(1.48, 0.) )
    plt.locator_params(nbins=6)
    #plt.show()
    return None

"SI F2, the basin of attraction for m=3"
committed_fraction = np.round(np.array([0.01, 0.01, 0.05]), 2)
committed_fraction = np.round(np.array([0, 0, 0]), 2)
committed_fraction = np.round(np.array([0.16, 0.15, 0.01]), 2)
committed_fraction = np.round(np.array([0.14, 0.13, 0.01]), 2)
committed_fraction = np.round(np.array([0.08, 0.07, 0.06]), 2)
committed_fraction = np.round(np.array([0.08, 0.02, 0.01]), 2)
committed_fraction = np.round(np.array([0.02, 0.01, 0]), 2)
committed_fraction = np.round(np.array([0.1, 0.04, 0.01]), 2)

p_fix = 0.16
#attractors = basin_attraction_number(committed_fraction)

#pqr_committed(number_opinion)
p_list = [0]
p_list = np.round(np.arange(0, 0.4, 0.02), 2)
p_list = [0.16]
for p_fix in p_list:
    #cross_section_committed(number_opinion, p_fix)
    #compare_reduce(4, p_fix)
    pass

pA_fix = 0.01

'''
for pA_fix in p_list:
    attractor_3D(number_opinion, pA_fix)
'''
pB_fix = 0.01
for pA_fix in p_list:
    for pB_fix in p_list:
        #attractor_value(number_opinion, pA_fix, pB_fix)
        pass

"Figure 2, change plot_ABC to show the fraction of any one the opinions v.s. p_A"
number_opinion = 3
p_B_plot = np.round(np.arange(0, 0.3, 0.03), 4 )
plot_ABC = 'B'
#pA_pcA_p(number_opinion, p_B_plot, plot_ABC)


"Figure 3, change number_opinion from 4 (Fig 3a) to 9 (Fig 3f)"
p_not_A = np.round(np.arange(0, 0.4, 0.04), 3)
plot_ABC = 'A_tilde'
number_opinion = 4
for number_opinion in [5, 6, 7, 8, 9]:
    for plot_ABC in ['A', 'B', 'A_tilde', 'C']:
        pA_pcA_p_approximation(number_opinion, p_not_A, plot_ABC)


"Figure 5, change number_opinion from 4 (Fig 5a) to 6 (Fig 5c)"
seed_list = np.arange(100)
normalization = 1
xplot = 'max'
sigma = 0.02
p_list = [0.02, 0.03, 0.04, 0.05]
number_opinion = 4
for p in p_list:
    #tipping_point_pmax_fluctuate(number_opinion, p, sigma, seed_list, normalization, xplot)
    pass

"Figure 5, change number_opinion from 4 (Fig 5d) to 6 (Fig 5f)"
xplot = 'sd'
sigma = 0.02
p_list = [0.02, 0.03, 0.04]
number_opinion = 6
for p in p_list:
    #tipping_point_pmax_fluctuate(number_opinion, p, sigma, seed_list, normalization, xplot)
    pass

approximation = 1
number_opinion_list = [3]
for number_opinion in number_opinion_list:
    #tipping_point(number_opinion, approximation)
    #tipping_point_two(number_opinion, approximation)
    pass
#tipping_point_fluctuate(number_opinion, p_list, sigma, seed_list, normalization)
#plt.show()

"plot figure 4b"
approximation = 1
number_opinion_list = [4, 5, 6, 7, 8, 9]
for number_opinion in number_opinion_list:
    #tipping_point(number_opinion, approximation)
    pass


number_opinion = 5
p_list = np.round(np.arange(0.005, 0.065, 0.001), 4)
#fluctuate_lowerbound(number_opinion, p_list)
sigma=0.02
p_list = [ 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]
#tipping_point_fluctuate(number_opinion, p_list, sigma, seed_list, normalization)
#turning_point_N(number_opinion_list, approximation)


#tipping_pmax(number_opinion, [0.005])
#triple_point(number_opinion, approximation)
#p_tipping = original_lowerbound(5, 0.03)
#pcA_psigma(4, 0.03)
#xA_pA_bounds(7, 0.06)
#xA_pA_bounds_gamma(6, 0.2)
gamma_list = np.round(np.arange(0.1, 1, 0.1), 3)
for gamma in gamma_list:
    #xA_pA_bounds_gamma_original(7, gamma)
    pass
p_list = np.round(np.arange(0.01, 0.1, 0.01), 3)
for p in p_list:
    #xA_pA_bounds_original(4, p)
    pass
number_opinion = 5
p_list = np.round(np.arange(0.01, 0.1, 0.01), 3)
#xs_pA_3D(number_opinion, p_list)
pA = np.round(0.12+1e-10, 11)
pAtilde_list = np.round(np.arange(0.01, 0.2, 0.02), 3)
number_opinion_list = np.arange(3, 50, 1)
for pAtilde in pAtilde_list:
    #xA_N_bounds(number_opinion_list, pA, pAtilde)
    pass
#xA_N_bounds_original(pA, pAtilde)
pA_list = np.round(np.arange(0.06, 0.17, 0.01) + 1e-10, 11)
#xA_N_approximation_four(pA_list, pAtilde_list)
p_sum_list = np.round(np.arange(0.06, 0.2, 0.02), 3)
#delta_x_delta_p(p_sum_list)
#xA_pA_two_opinion(2)
#tipping_point_continuous(3, 1)
for number_opinion in np.arange(5, 11, 1):
    #peak_A(number_opinion)
    pass
plt.show()
