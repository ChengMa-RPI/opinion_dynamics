""" transfer files between local machine and remote server"""
import paramiko
import os 
import numpy as np 

client = paramiko.SSHClient()
client.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
# client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

client.connect('ganxis3.nest.rpi.edu', username='mac6', password='woods*score&sister')

def transfer_files(directory, filenames):

    server_des = '/home/mac6/RPI/research/opinion_dynamics/data/' 
    local_des = '/home/mac/RPI/research/opinion_dynamics/data/'
    if not os.path.exists(local_des):
        os.makedirs(local_des)
    sftp = client.open_sftp()
    if '/' in directory:
        if not os.path.exists(local_des + directory):
            os.makedirs(local_des + directory)
        filenames = sftp.listdir(server_des+directory) 
    for i in filenames:
        sftp.get(server_des + directory + i, local_des + directory +i)
        #sftp.put(local_des + directory +i, server_des + directory + i)
    sftp.close()


directory =  '../data/'
number_opinion = 6
if not os.path.exists(directory):
    os.makedirs(directory)


#filenames = directory +  f'num_opinion={number_opinion}_stable.csv'
#filenames = directory +  f'num_opinion={number_opinion}_oneuncommitted.csv'
filenames = directory +  f'num_opinion={number_opinion}_oneuncommitted_approximation.csv'
#transfer_files('', [filenames])

number_opinion = 4
normalization = 1
sigma_list = [0.005, 0.01, 0.02]
p_list = [ 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]
sigma_list = [0.02]
p_list = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065]
seed_list = np.arange(100).tolist()
directory += f'num_opinion={number_opinion}_absolute/'
if not os.path.exists(directory):
    os.makedirs(directory)
for p in p_list:
    for sigma in sigma_list:
        for seed in seed_list:
            if normalization: 
                filenames = directory  + f'oneuncommitted_p={p}_sigma={sigma}_seed={seed}_normalization.csv'
            else:
                filenames = directory + f'oneuncommitted_p={p}_sigma={sigma}_seed={seed}.csv'
            #transfer_files('', [filenames])

number_opinion = 6
directory = f'num_opinion={number_opinion}_lowerbound/'
p_list = [0.005,  0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065]
p_list = np.round(np.arange(0.005, 0.07, 0.001), 4)
for p in p_list:
    filenames = directory  + f'oneuncommitted_p={p}.csv'
    #transfer_files('', [filenames])

number_opinion = 5
directory = f'num_opinion={number_opinion}_original_lowerbound/'
p_list = [0.045]
for p in p_list:
    filenames = directory  + f'p={p}.csv'
    #transfer_files('', [filenames])

number_opinion = 5
directory = f'num_opinion={number_opinion}_original/'
p_list = np.round(np.arange(0.01, 0.1, 0.01), 3)
for p in p_list:
    filenames = directory  + f'p={p}_random.csv'
    #transfer_files('', [filenames])

number_opinion = 7
directory = f'../data/num_opinion={number_opinion}_oneuncommitted_approximation_four/'
gamma_list = np.round(np.arange(0.01, 0.1, 0.01), 3)
gamma_list = [0.7, 0.8, 0.9]
for gamma in gamma_list:
    filenames = directory  + f'gamma={gamma}.csv'
    #transfer_files('', [filenames])

directory = f'../data/Nchange_original/'
N_list = np.arange(3, 9, 1)
pA = np.round(0.1 + 1e-10, 11)
pAtilde = np.round(0.14, 3)
p0 = pAtilde/(N_list - 2)
index = np.where(pA > p0)[0]
N_index = N_list[index]
for number_opinion in N_index:
    filenames = directory + f'num_opinion={number_opinion}/pA={pA}_pAtilde={pAtilde}_random.csv'
    #transfer_files('', [filenames])

directory = f'../data/Nchange_oneuncommitted_approximation_three/'

N_list = np.arange(51, 87, 1)
for number_opinion in N_list:
    filenames = directory + f'num_opinion={number_opinion}.csv'
    #transfer_files('', [filenames])

number_opinion = 22
directory = f'actual_simulation/number_opinion={number_opinion}/'

N_list = np.array([5000, 10000, 50000, 100000])
N_list = np.array([1000, 5000, 10000, 50000, 100000])
interaction_number_list = N_list * 100
pA_list = np.array([0.12, 0.14, 0.16, 0.18])
p_list = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
for N, interaction_number in zip(N_list, interaction_number_list):
    for pA in pA_list:
        for p in p_list:
            file_directory = directory + f'N={N}_interaction_number={interaction_number}_pA={pA}_p={p}/'
            #transfer_files(file_directory, [])

pAtilde_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
number_opinion_list = np.arange(3, 9, 1)
ode = 0
"listener-only version"
for pAtilde in pAtilde_list:
    for number_opinion in number_opinion_list:
        if ode == 1:
            file_directory = '../data/listener_only/ode/pA_critical_N/'
        else:
            file_directory = '../data/listener_only/recursive/pA_critical_N/'
        file_names = file_directory + f'pAtilde={pAtilde}_N={number_opinion}.csv'
        #transfer_files('', [file_names])



pAtilde_list = [0.18]
number_opinion_list = np.arange(3, 8, 1)
ode = 1
alpha_list = [10]
seed_list = np.arange(0, 30, 1)
for pAtilde in pAtilde_list:
    for number_opinion in number_opinion_list:
        for seed in seed_list:
            for alpha in alpha_list:
                if ode == 1:
                    file_directory = '../data/listener_only/ode/pA_critical_N_fluctuation_dirichlet/'
                else:
                    file_directory = '../data/listener_only/recursive/pA_critical_N_fluctuation_dirichlet/'
                file_names = file_directory + f'pAtilde={pAtilde}_N={number_opinion}_alpha={alpha}_seed={seed}.csv'
                #transfer_files('', [file_names])

number_opinion = 2

directory = f'actual_simulation/onecommitted/number_opinion={number_opinion}/'

N_list = np.array([1000000])
interaction_number_list = N_list * 100
pA_list = np.array([0.3])
for N, interaction_number in zip(N_list, interaction_number_list):
    for pA in pA_list:
        file_directory = directory + f'N={N}_interaction_number={interaction_number}_pA={pA}/'
        #transfer_files(file_directory, [])


number_opinion = 2
approx_integer = 'floor'
directory = f'actual_simulation/number_opinion={number_opinion}/approx_integer=' + approx_integer + '/'
pA = 0.3
p0 = 0
N_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
N_list = [1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
N_list = [20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
N_list = [50000, 60000, 70000, 80000, 90000, 100000]
N_list = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
switch_direction = 'B-A'
for N in N_list:
    file_directory = directory + f'N={N}_pA={pA}_pB={p0}_switch_direction=' + switch_direction + '/'
    #transfer_files(file_directory, [])




N = 1000
N = 10000
N = 100000
interaction_number = N * 1000

network_type = 'SF'
d_list = [[2.1, 0, 2], [2.5, 0, 3], [3.5, 0, 4]]
net_seed_list = [[5, 5], [98, 98], [79, 79]]



network_type = 'ER'
d_list = [30000, 40000, 80000, 160000]
net_seed_list = [1, 0, 0, 0]

network_type = 'complete'
d_list = [0]
net_seed_list = [0]


pA_list = np.round(np.arange(0.01, 0.16, 0.01), 2)
number_opinion_list = [2] * 8 + [3] * 8  + [4] * 2 + [5] * 4 + [6] * 8 + [7] * 2
p_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.01, 0.02, 0.005, 0.01, 0.015, 0.02, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.014, 0.016, 0.005, 0.01]


pA_List = [0.01, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
number_opinion_list = [5]
p_list = [0.01]

pA_list = np.round(np.arange(0.01, 0.16, 0.01), 2)
number_opinion_list = [ 2, 3, 4, 7]
p_list = [ 0.06, 0.03, 0.02, 0.01]


for d, net_seed in zip(d_list, net_seed_list):
    for pA in pA_list:
        for p, number_opinion in zip(p_list, number_opinion_list):
            file_directory = f'../data/' + network_type + f'/N={N}_d={d}_netseed={net_seed}/actual_simulation/number_opinion={number_opinion}/interaction_number={interaction_number}_pA={pA}_p={p}/'
            transfer_files(file_directory, [])


