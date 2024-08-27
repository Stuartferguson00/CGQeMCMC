import sys
import os
dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(dir))
sys.path.insert(0, root_dir)

import mpmath as mp
import numpy as np

from cgqemcmc.energy_models import IsingEnergyFunction, Exact_Sampling
from cgqemcmc.basic_utils import *

import pickle
import gc
import scipy



import arviz as az

#quit()

import matplotlib.pyplot as plt





    
def fill_missing_values(x, y, x_min, x_max, num_samples = 1000):
    
    #Function to return a "thin" Markov chain where only accepted updates are stored (for efficient storage) to a more "full" markov chain
    
    x = np.array(x)
    y = np.array(y)
    valid_indices = np.where(x != None)[0]

    # Remove None values from x and y using boolean indexing
    x = x[valid_indices]
    y = y[valid_indices]
    #print("number of steps accepted")
    #print(len(y))
    
    if x_min <1:
        x_min = 1


    #full_x = np.logspace(max(np.log10(x_min),0), np.log10(x_max + 1), 100, dtype = int)
    #full_x[0] = 0
    #full_x = np.linspace(x_min, x_max, num_samples, dtype = int)

    full_x = np.logspace(max(np.log10(x_min),0), np.log10(x_max + 1), num_samples, dtype = int)
    full_x = np.unique(full_x)
    full_y = np.zeros_like(full_x, dtype = type(y[0]))
    

    
    count = 0
    current_y = y[count]
    for i, x_ in enumerate(full_x):
        
        if x_ == 0:
            pass
        elif count>=len(x):
            pass
        elif x_ >= x[count]:
            if count+1>=len(x):
                pass
            elif x_ < x[count+1]:
                current_y = y[count]
                count += 1
            else:
                not_found = True
                while not_found:
                    if x_ >= x[count]:
                        if count+1>=len(x):
                            not_found = False
                        elif x_ < x[count+1]:
                            current_y = y[count]
                            count += 1
                            not_found = False
                        else:
                            count += 1
                    else:
                        print("something failed")
                        
        full_y[i] = current_y

    return full_x, full_y


def hamming_distance(bin_str1, bin_str2):

    # Calculate Hamming distance
    distance = sum(bit1 != bit2 for bit1, bit2 in zip(bin_str1, bin_str2))

    return distance

def find_largest_distance_away(state_list):
    hamming_dists = []
    energy_diffs = []
    for m, state in enumerate(state_list):
        if m == 0:
            n = state.bitstring
            e = state.energy
        else:
            hamming_dists.append(hamming_distance(n,state.bitstring))
            energy_diffs.append(abs(e-state.energy))
            #if count >100:
            #    #print(n)
            #    count = 0
            #count +=1
    return np.max(hamming_dists)

def get_results_dir_n_spins(results_dir):

    l_results_dir = list(results_dir)
    if n_spins >=100:
        l_results_dir[-11] = str(n_spins)[0]
        l_results_dir[-10] = str(n_spins)[1]
        l_results_dir[-9] = str(n_spins)[2]
    elif n_spins >=10:
        l_results_dir[-10] = str(n_spins)[0]
        l_results_dir[-9] = str(n_spins)[1]
    else:
        l_results_dir[-9] = str(n_spins)
    l_results_dir = list(l_results_dir)
    results_dir = ''.join(l_results_dir)

    l_results_dir = list(results_dir)


    results_dir = ''.join(l_results_dir)

    return results_dir

def get_results_dir_m_q(results_dir, m_q, st = "rl"):
    l_results_dir = list(results_dir)
    l_results_dir[-7] = str(m_q/n_spins)[2]
    try:
        l_results_dir[-6] = str(m_q/n_spins)[3]
    except:
        l_results_dir[-6] = str(0)
    try:
        l_results_dir[-5] = str(m_q/n_spins)[4]
    except:
        l_results_dir[-5] = str(0)

    l_results_dir[-14] = st[0]
    l_results_dir[-13] = st[1]
    
    results_dir = ''.join(l_results_dir)
    
    return results_dir

def get_results_dir_class(results_dir, code = "uo"):
    l_results_dir = list(results_dir)
    l_results_dir[-7] = str(0)
    l_results_dir[-6] = str(0)
    l_results_dir[-5] = str(0)

    l_results_dir[-14] = code[0]
    l_results_dir[-13] = code[1]
    
    results_dir = ''.join(l_results_dir)
    
    return results_dir


    







def unpickle(results_dir):
    gc.disable()
    
    fileObj = open(results_dir, 'rb')
    results_list = pickle.load(fileObj)
    fileObj.close()
    gc.enable()

    all_energies = []
    all_states = []
    hops = []

    for i in range(len(results_list)):
        accepted_energies, accepted_positions, accepted_states = results_list[i]

        
        accepted_energies = np.array(accepted_energies)
        accepted_states = np.array(accepted_states)
        accepted_positions = np.array(accepted_positions)
        accepted_positions[accepted_positions == None] = 0

        
        all_energies.append(accepted_energies)
        all_states.append(accepted_states)
        hops.append(accepted_positions)
        
    return all_energies, all_states, hops



def plot_energies(all_energies, hops, ax, label, color, sampled = False):
    #sampled is whether the data is from sampling a markov chain or recording accepted energies
    # sampled = False means that "fill_missing_values" must be called.

    full_energies = []
    for i in range(len(all_energies)):
        if not sampled:
            full_hops, full_energies_ = fill_missing_values(hops[i], all_energies[i], auto_burnin, 10**np.ceil(np.log10(np.max(np.array(hops[0])[1:]))))
        else:
            hops[i][hops[i]==0] = 1
            full_hops = hops[i]
            full_energies_ = all_energies[i]
        full_energies.append(full_energies_)
        if plot_individuals:
            if i >0:
                label_ = None
            else:
                label_ = label +" individual"
            ax[1].plot(full_hops, full_energies_, label = label_, color = color, alpha = 0.3)
            ax[0].plot(full_hops, full_energies_, label = label_, color = color, alpha = 0.3)
            
            pass
            
    full_hops= np.array(full_hops)
    full_energies = np.array(full_energies)
    ax[0].plot(full_hops, np.mean(full_energies,axis = 0), label = label, color = color)
    ax[1].plot(full_hops, np.mean(full_energies,axis = 0), label = label, color = color)



    #fig.suptitle("Thermalisation of MCMC for "+str(n_spins)+" spins")
    ax[1].set_xscale("log")
    ax[0].set_xscale("log")
    
    full_hops = np.array(full_hops)
    full_energies = np.array(full_energies)

    
    return full_hops, full_energies

def plot_low_e(max):
    print("plot_low_e")
    if n_spins <= 16 or got_vals:
        for i, le in enumerate(lowest_energies[:10]):
            color = interpolate_color("springgreen", "darkblue", float((i)/len(lowest_energies)))
            if i ==0:
                ax[1].plot([0,max+max/50],[le,le], color = color, ls = "--", alpha = 0.1, label = "Lowest energy levels")#, label = str(i)+" excited_sate"
                ax[0].plot([0,max+max/50],[le,le], color = color, ls = "--", alpha = 0.1, label = "Lowest energy levels")#, label = str(i)+" excited_sate"

            else:
                ax[1].plot([0,max+max/50],[le,le],  color = color, ls = "--", alpha = 0.1)#label = str(i)+" excited_sate",
                ax[0].plot([0,max+max/50],[le,le], color = color, ls = "--", alpha = 0.1)#, label = str(i)+" excited_sate"
            

        zoom_y_min, zoom_y_max = np.min(lowest_energies)-1, np.max(lowest_energies)-np.max(lowest_energies)*0.1
        #zoom_x_min, zoom_x_max = 0, np.max(list(itertools.chain.from_iterable(Q_hops)))


        #ax[1].set_xlim(zoom_x_min, zoom_x_max)
        ax[1].set_ylim(zoom_y_min, zoom_y_max)



    

        lowest_energy = np.min(lowest_energies)
        highest_probs = highest_prob_states
    

        #avg_energy = np.average(lowest_energies, weights = highest_probs/100)
        print(lowest_energies)
        print(highest_probs)
        avg_energy = np.sum(lowest_energies*highest_probs)
    
        
            

        print("sum of probs (if much lower than 100, avg_energy will be innacurate)")
        print(np.sum(highest_probs))

        print("Average energy:", avg_energy)
        print("max", max)
        ax[0].plot([0,max+max/50],[avg_energy,avg_energy], color = "k", label = "Exact average energy")
        ax[1].plot([0,max+max/50],[avg_energy,avg_energy], color = "k", label = "Exact average energy")




def compute_rhat(chain_values, index_list=None):
    if index_list is not None:
        rhats = []
        for i in index_list:
            if i >= 10:
                # Compute R-hat value using ArviZ
                rhats.append(az.rhat(chain_values[:, (i//10):i])-1)
        return np.array(rhats)
    else:
        return az.rhat(chain_values)

def compute_mcse(chain_values, index_list = None):
    if index_list is not None:
        mcse = []
        for i in index_list:
            if i >= 10:
                # Compute R-hat value using ArviZ
                mcse.append(az.mcse(chain_values[:, (i//10):i]))
        return np.array(mcse)
    else:
        return az.mcse(chain_values)

def compute_ess(chain_values, index_list=None):
    if index_list is not None:
        ess = []
        for i in index_list:
            if i >= 10:
                # Compute R-hat value using ArviZ
                ess.append(az.ess(chain_values[:, (i//10):i]))
        return np.array(ess)
    else:
        ess = az.ess(chain_values)
        return ess



def get_mags(all_energies,all_states, hops, full_hops, sampled):
    print("getting mags")
    mags_l = []
    for i in range(len(all_energies)):
        mags = []
        for k in range(len(hops[i])):
            try:
                mags.append(magnetization_of_state(all_states[i][k]))
            except:
                mags.append(magnetization_of_state(all_states[i][k].bitstring))
        mags_l.append(mags)

    #fill out
    if sampled:
        mags_full = mags_l
    else:
        mags_full = []
        for i in range(len(mags_l)):
            _, mags_full_ = fill_missing_values(hops[i], mags_l[i], full_hops[0], full_hops[-1])
            print(len(mags_full_))
            mags_full.append(mags_full_)
        mags_full = np.array(mags_full)
    return mags_full

def get_unique_counts(full_energies):
    #get unique counts
    unique_dict = {}

    for i in range(len(full_energies)):
        unique, count = np.unique(full_energies[i], return_counts = True)
        #print(unique, count)

        
        for k, u in enumerate(unique):
            try:
                unique_dict[u] += 1#float(count[k])
            except:
                unique_dict[u] = 1#float(count[k])
                
                
    tot_sum_counts = np.sum([len(i) for i in full_energies])

    if tot_sum_counts>1:
        #normalise and turn to epercentage
        for key in unique_dict:
            unique_dict[key] = float(unique_dict[key])
            #unique_dict[key] *= float(100/tot_sum_counts)

        highest_values = dict(sorted(unique_dict.items(), key=lambda item: item[1], reverse=True)[:5])

    return highest_values

def thin_stuff(lst, thin = 10):
    #lst is list of things to thin out
    #thin is how often to thin (ie thin = 10 means keep every 10th element)
    lst_ = []
    for l in lst:
        l = np.array(l)
        lst_.append(np.concatenate((l[:,:10],l[:,10::thin]), axis = 1))
    return lst_




def estimate_partition_function(energies, Temp):
    Z = 0
    for e in np.flip(energies):
        Z += mp.exp(-e/Temp) 
    return Z

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# START
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

n_spins = 16

temp = float(0.1)#float(1)
auto_burnin = 0

plot_first = True
plot_cums = True
plot_individuals = False
plot_convergence_analysis = False


do_local = True
l_sampled = True
thin_l = 100
l_color = "green"

do_uniform = True
u_sampled = True
thin_u = 100
u_color = "orange"

do_quantum_Q = False   
Q_sampled =   True
thin_Q = 1
Q_color = "purple"


do_quantum_q = True
q_sampled = True
thin_q = 10
q_color = "r"

do_quantum_q_2 = False  
q_2_sampled = True
thin_q_2 = 10
q_2_color = "pink"

do_quantum_nms = False  
nms_sampled = True
thin_nms = 10
nms_color = "lightblue"


do_quantum_dp = True  
dp_sampled = True
thin_dp = 10
dp_color = "k"

do_quantum_dp_2 = False  
dp_2_sampled = True
thin_dp_2 = 10
dp_2_color = "blue"


m_Q = n_spins
m_q = int(np.sqrt(n_spins))
m_q_2 = n_spins//3
m_q_nms = int(np.sqrt(n_spins))
m_q_dp = int(np.sqrt(n_spins))
m_q_dp_2 = int(n_spins)


t_str = str(temp)
t_str = t_str.replace(".", "_")
if len(t_str)>6:
    t_str = t_str[:6]



model_dir = dir+'/Models/000.obj'




results_dir = dir+'/Results/'+t_str+'/oo_000_000.obj'
Q_results_dir = dir+'/Results/'+t_str+'/oo_000_000.obj'
nms_results_dir = dir+'/Results/'+t_str+'/non_ms_oo_000_000.obj'
dp_results_dir = dir+'/Results/'+t_str+'/oo_000_000.obj' 




#change file names for easy file organisation
l_model_dir = list(model_dir)
if n_spins >=100:
    l_model_dir[-7] = str(n_spins)[0]
    l_model_dir[-6] = str(n_spins)[1]
    l_model_dir[-5] = str(n_spins)[2]

    
elif n_spins >=10:
    l_model_dir[-6] = str(n_spins)[0]
    l_model_dir[-5] = str(n_spins)[1]
else:
    l_model_dir[-5] = str(n_spins)

model_dir = ''.join(l_model_dir)





fileObj = open(model_dir, 'rb')
model_list = pickle.load(fileObj)
fileObj.close()
if len(model_list) == 1:
    m_l = []
    for i in range(500):
        m_l.append(model_list[0])
    model_list = m_l
    
    
    
    
    









    
try:
    
    print("try1")
    
    l_model_dir = list(model_dir)
    l_model_dir = l_model_dir[:-4] 
    #l_model_dir[-8] = "c"

    analysis_model_dir = ''.join(l_model_dir) + "_"+t_str +".obj"
    #print("print(analysis_model_dir)")
    #print(analysis_model_dir)
    fileObj = open(analysis_model_dir, 'rb')
    outp = pickle.load(fileObj)
    fileObj.close()
    
    highest_prob_states,lowest_energies,minima_mags,exact_mag = outp
    print("got to this one")
    print(lowest_energies)
    print(highest_prob_states)
    
except:
    print("except1")

    if n_spins <= 16:
    
        #get minima states

        #assumes all models are the same
        m = model_list[0]
        ES = Exact_Sampling(m, 1/temp)
        boltz_dict = ES.get_boltzmann_distribution(1/temp,sorted = True, save_distribution=True)
        min_s_bitstrings = np.array(list(boltz_dict.keys()))[:30]
        min_s_mu = np.array(list(boltz_dict.values()))[:30]
        lowest_energies, degeneracy = m.get_lowest_energies(30)
        lowest_energies_2 = [m.get_energy(s) for s in min_s_bitstrings]

        lowest_energies = np.array(lowest_energies)




        #print("Highest prob states (percentage)")
        #np.set_printoptions(suppress=True,precision=2)
        highest_prob_states = min_s_mu
        
        #print(min_s_mu*100)
        #np.set_printoptions(suppress=True,precision=6)
        #print("lowest_energies and degeneracy")
        #print(lowest_energies, degeneracy)
        #print("lowest_energies_2")
        #print(lowest_energies_2)
        
        #the variable required for spec gap thermalisation equation
        min_s_mu_s = np.array(list(boltz_dict.values()))[-1]
        #print("min_s_mu_s, the variable required for spec gap thermalisation equation")
        #print(min_s_mu_s)
        #for m in np.array(list(boltz_dict.values())):
        #    print(m)

        minima_mags = []
        print("low state bitstrings")
        for i in range(len(min_s_mu)):
            minima_mags.append(magnetization_of_state(min_s_bitstrings[i]))
            print(min_s_bitstrings[i])
        #print("Magnetisation of minima")
        
        #print(minima_mags)

        #get actual_magnetisation


        mag_dict = dict_magnetization_of_all_states(list(boltz_dict.keys()))
        exact_mag = np.sum(np.array(list(mag_dict.values()))*np.array(list(boltz_dict.values())))
        #print("exact mag is: "+str(exact_mag))
        #print(mag_dict[min_s_bitstrings[0]])




        second_exact_mag =  np.sum(np.array(list(mag_dict.values()))[2:]*np.array(list(boltz_dict.values()))[2:])
        #print("exact mag without ground state is: "+str(second_exact_mag))


        fileObj = open(analysis_model_dir, 'wb')
        pickle.dump([highest_prob_states,lowest_energies,minima_mags,exact_mag, min_s_mu_s],fileObj)
        fileObj.close()
        
        
        got_vals = True
        
    else:
        try:
            print("trying this")
            lowest_energies = np.loadtxt(dir+"/Models/low_neg_energies_0"+str(n_spins)+".txt")
            print("first worked")
            low_states = np.loadtxt(dir+"/Models/low_neg_states_0"+str(n_spins)+".txt")
            print("this worked")
            
            lowest_energies = lowest_energies[:30]
            low_states = low_states[:30]
            
            minima_mags = []
            for i in range(len(low_states)):
                state = low_states[i,:]
                state[state == -1] = 0
                bitstring = "".join(str(int(i)) for i in state)
                minima_mags.append(magnetization_of_state(bitstring))
            minima_mags = np.array(minima_mags)
            highest_prob_states = np.nan
            
            
            
            partition_func = estimate_partition_function(lowest_energies, temp)
            print("partition_func: "+str(partition_func))
            exp_ = []
            for e in lowest_energies:
                exp_.append(mp.exp(-e / temp))
            print("exp_: "+str(exp_))
            highest_prob_states = []
            
            for e in exp_:
                highest_prob_states.append(e / partition_func)
            #highest_prob_states = exp_ / partition_func
            print("highest_prob_states")
            print(highest_prob_states)
            avg_energy = np.sum(lowest_energies*highest_prob_states)
            exact_mag = np.sum(minima_mags*highest_prob_states)
            print("Average energy:", avg_energy)
            

            print("completed, shouldnt go to eventually")
            got_vals = True
        except:
            print("eventually did this")
            got_vals = False
        
        
            
if n_spins <= 16 or got_vals:
    print("exact mag is: "+str(exact_mag))
    np.set_printoptions(suppress=True,precision=2)
    print("highest_prob_states")
    print(highest_prob_states)
    np.set_printoptions(suppress=True,precision=6)
    print("lowest_energies")
    print(lowest_energies)
    print("minima_mags")
    print(minima_mags)
    #print("min_s_mu_s -- the thing required for spec gap thermalisation equation")
    #print(min_s_mu_s)

m = model_list[0]











#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#quantum
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


fig, ax = plt.subplots(1,2)#,figsize=(8, 8))

if do_quantum_Q:
    Q_results_dir = get_results_dir_n_spins(Q_results_dir)
    Q_results_dir = get_results_dir_m_q(Q_results_dir, m_Q)
    Q_all_energies, Q_all_states, Q_hops = unpickle(Q_results_dir)
    if thin_Q>1:
        Q_all_energies, Q_all_states, Q_hops = thin_stuff([Q_all_energies, Q_all_states, Q_hops],thin_Q)
    Q_full_hops, Q_full_energies = plot_energies(Q_all_energies, Q_hops, ax, str(int(m_Q))+" qubits multiple sample", Q_color, sampled = Q_sampled)

if do_quantum_q:
    q_results_dir = get_results_dir_n_spins(results_dir)
    q_results_dir = get_results_dir_m_q(q_results_dir, m_q)
    q_all_energies, q_all_states, q_hops = unpickle(q_results_dir)
    if thin_q>1:
        q_all_energies, q_all_states, q_hops = thin_stuff([q_all_energies, q_all_states, q_hops],thin_q)
    q_full_hops, q_full_energies = plot_energies(q_all_energies, q_hops, ax, str(int(m_q))+" qubits multiple sample", q_color, sampled = q_sampled)
    q_highest_values = get_unique_counts(q_all_energies)
    print("highets values q = "+str(q_highest_values))
    
    
if do_quantum_q_2:
    q_2_results_dir = get_results_dir_n_spins(results_dir)
    q_2_results_dir = get_results_dir_m_q(q_2_results_dir, m_q_2)
    q_2_all_energies, q_2_all_states, q_2_hops = unpickle(q_2_results_dir)
    if thin_q_2>1:
        q_2_all_energies, q_2_all_states, q_2_hops = thin_stuff([q_2_all_energies, q_2_all_states, q_2_hops],thin_q_2)
    q_2_full_hops, q_2_full_energies = plot_energies(q_2_all_energies, q_2_hops, ax, str(int(m_q_2))+" qubits multiple sample", q_2_color, sampled = q_2_sampled)

if do_quantum_nms:
    nms_results_dir = get_results_dir_n_spins(nms_results_dir)
    nms_results_dir = get_results_dir_m_q(nms_results_dir, m_q_nms)
    nms_all_energies, nms_all_states, nms_hops = unpickle(nms_results_dir)
    if thin_nms>1:
        nms_all_energies, nms_all_states, nms_hops = thin_stuff([nms_all_energies, nms_all_states, nms_hops],thin_nms)
    nms_full_hops, nms_full_energies = plot_energies(nms_all_energies, nms_hops, ax, str(int(m_q_nms))+" qubits improved local group", nms_color, sampled = nms_sampled)

if do_quantum_dp:
    dp_results_dir = get_results_dir_n_spins(dp_results_dir)
    dp_results_dir = get_results_dir_m_q(dp_results_dir, m_q_dp, "dp")
    dp_all_energies, dp_all_states, dp_hops = unpickle(dp_results_dir)
    if thin_dp>1:
        dp_all_energies, dp_all_states, dp_hops = thin_stuff([dp_all_energies, dp_all_states, dp_hops],thin_dp)
    dp_full_hops, dp_full_energies = plot_energies(dp_all_energies, dp_hops, ax, str(int(m_q_dp))+" qubits noise", dp_color, sampled = dp_sampled)


if do_quantum_dp_2:
    dp_2_results_dir = get_results_dir_n_spins(dp_results_dir)
    dp_2_results_dir = get_results_dir_m_q(dp_2_results_dir, m_q_dp_2, "dp")
    dp_2_all_energies, dp_2_all_states, dp_2_hops = unpickle(dp_2_results_dir)
    if thin_dp_2>1:
        dp_2_all_energies, dp_2_all_states, dp_2_hops = thin_stuff([dp_2_all_energies, dp_2_all_states, dp_2_hops],thin_dp_2)
    dp_2_full_hops, dp_2_full_energies = plot_energies(dp_2_all_energies, dp_2_hops, ax, str(int(m_q_dp_2))+" qubits noise", dp_2_color, sampled = dp_2_sampled)
    
if do_local:
    l_results_dir = get_results_dir_n_spins(results_dir)
    l_results_dir = get_results_dir_class(l_results_dir, code = "lo")
    l_all_energies, l_all_states, l_hops = unpickle(l_results_dir)

    if thin_l>1:
        l_all_energies, l_all_states, l_hops = thin_stuff([l_all_energies, l_all_states, l_hops],thin_l)
    l_full_hops, l_full_energies = plot_energies(l_all_energies, l_hops, ax, "local", l_color, sampled = l_sampled)

    l_highest_values = get_unique_counts(l_all_energies)
    print("highets values l = "+str(l_highest_values))
    
if do_uniform:
    u_results_dir = get_results_dir_n_spins(results_dir)
    u_results_dir = get_results_dir_class(u_results_dir, code = "uo")
    u_all_energies, u_all_states, u_hops = unpickle(u_results_dir)
    u_hops = np.array(u_hops)

    if thin_u>1:
        u_all_energies, u_all_states, u_hops = thin_stuff([u_all_energies, u_all_states, u_hops],thin_u)
    u_full_hops, u_full_energies = plot_energies(u_all_energies, u_hops, ax, "uniform", u_color, sampled = u_sampled)

    u_highest_values = get_unique_counts(u_all_energies)
    print("highets values u = "+str(u_highest_values))

if got_vals:
    try:
        plot_low_e(np.max(l_full_hops))
    except:
        plot_low_e(np.max(q_full_hops))




#handles, labels = fig.gca().get_legend_handles_labels()
#unique_labels = []
#unique_handles = []
#for i, label in enumerate(labels):
#    if label not in unique_labels:
#        unique_labels.append(label)
#        unique_handles.append(handles[i])
#ax[0].legend(unique_handles, unique_labels)


ax[0].legend()

if plot_first:
    #ax[0].text(10,-2,"T = "+str(temp))

    fig.suptitle("Thermalisation of MCMC for "+str(n_spins)+" spins | T = "+str(temp))
    fig.supxlabel("Steps")
    fig.supylabel("Energy")
    
    
    fig.delaxes(ax[1])
    ax[0].set_position([0.1, 0.1, 0.8, 0.8])
    
    
    #ax[0].set_title("Full Thermalisation")
    #ax[1].set_title("Low energy region")

    
    #if got_vals:
    #    ax[1].set_ylim(np.min(lowest_energies)-3,np.max(lowest_energies)+3)
    #else:
    #    ax[1].set_ylim(np.min(l_all_energies)-3,np.min(l_all_energies)+10)

    
    plt.show()
else:
    plt.close()















# Assuming you have defined the necessary variables and lists before this section

# Calculate the logspace for hops

# PLOT CUMULATIVES
print("PLOTTING CUMULATIVES")




def plot_cummulatives(mags_l, _hops,  axs, label =  "Uniform", color = "orange", plot = True):
    
    mags_cumulative_l = []
    for j, mags in enumerate(mags_l):
        mags_cumulative = []
        for i in range(1,len(_hops)):
            mags_cumulative.append(np.sum(mags[:i]) / (len(mags[:i])))

        
        mags_cumulative_l.append(mags_cumulative)
        #axs[0].plot(_hops, mags_cumulative, color=color, alpha=0.1)
        
    _hops = _hops[1:]
    yerrs = scipy.stats.sem(mags_cumulative_l, axis=0)
    mean = np.mean(mags_cumulative_l, axis=0)
    
    print(_hops.shape)    
    #for y in yerrs:
    #    if type(y) is not float:
    #        print(y)
    #        print(type(y))
    if plot:
        axs[0].plot(_hops, mean, label=label, color=color, alpha=1)
    

    #axs[0].fill_between(_hops[1:], mean[1:] - yerrs[1:], mean[1:] + yerrs[1:], alpha=0.2, color=color)
    
    mags_cumulative_l = np.array(mags_cumulative_l)
    return mags_cumulative_l
    



if plot_cums:
    # Plotting subplot 1: u_mags_cumulative vs. hops
    fig_cum, axs_cum = plt.subplots(2, 1)
else:
    axs_cum = None



if do_quantum_Q:
    Q_mags_full = get_mags(Q_all_energies, Q_all_states, Q_hops, Q_full_hops, sampled = Q_sampled)
    Q_mags_cumulative_l = plot_cummulatives(Q_mags_full, Q_full_hops, axs_cum, label =  str(int(m_Q))+" qubits multiple sample", color = Q_color,plot= plot_cums)
    Q_mags_mean = np.mean(Q_mags_cumulative_l, axis=0)
    if got_vals:
        if plot_cums:
            Q_distances = [abs(val - exact_mag) for val in Q_mags_mean]
            axs_cum[1].plot(Q_full_hops[1:], Q_distances, color=Q_color)

if do_quantum_q:
    q_mags_full = get_mags(q_all_energies, q_all_states, q_hops, q_full_hops, sampled = q_sampled)
    q_mags_cumulative_l = plot_cummulatives(q_mags_full, q_full_hops, axs_cum, label =  str(int(m_q))+" qubits multiple sample", color = q_color,plot = plot_cums)
    q_mags_mean = np.mean(q_mags_cumulative_l, axis=0)
    
    if got_vals:
        if plot_cums:
            q_distances = [abs(val - exact_mag) for val in q_mags_mean]
            axs_cum[1].plot(q_full_hops[1:], q_distances, color=q_color)
            
            
if do_quantum_q_2:
    q_2_mags_full = get_mags(q_2_all_energies, q_2_all_states, q_2_hops, q_2_full_hops, sampled = q_2_sampled)
    q_2_mags_cumulative_l = plot_cummulatives(q_2_mags_full, q_2_full_hops, axs_cum, label =  str(int(m_q_2))+" qubits multiple sample", color = q_2_color,plot = plot_cums)
    q_2_mags_mean = np.mean(q_2_mags_cumulative_l, axis=0)
    
    if got_vals:
        if plot_cums:
            q_2_distances = [abs(val - exact_mag) for val in q_2_mags_mean]
            axs_cum[1].plot(q_2_full_hops[1:], q_2_distances, color=q_2_color)
            
            
            
if do_quantum_nms:
    nms_mags_full = get_mags(nms_all_energies, nms_all_states, nms_hops, nms_full_hops, sampled = nms_sampled)
    nms_mags_cumulative_l = plot_cummulatives(nms_mags_full, nms_full_hops, axs_cum, label =  str(int(m_q_nms))+" qubits multiple sample", color = nms_color,plot = plot_cums)
    nms_mags_mean = np.mean(nms_mags_cumulative_l, axis=0)
    
    if got_vals:
        if plot_cums:
            nms_distances = [abs(val - exact_mag) for val in nms_mags_mean]
            axs_cum[1].plot(nms_full_hops[1:], nms_distances, color=nms_color)

if do_local:
    l_mags_full = get_mags(l_all_energies, l_all_states, l_hops, l_full_hops, sampled = l_sampled)
    l_mags_cumulative_l = plot_cummulatives(l_mags_full, l_full_hops, axs_cum, label =  "Local", color = l_color,plot = plot_cums)
    l_mags_mean = np.mean(l_mags_cumulative_l, axis=0)
    
    if got_vals:
        if plot_cums: 
            l_distances = [abs(val - exact_mag) for val in l_mags_mean]
            axs_cum[1].plot(l_full_hops[1:], l_distances, color=l_color)
    
if do_uniform:
    u_mags_full = get_mags(u_all_energies, u_all_states, u_hops, u_full_hops, sampled = u_sampled)
    u_mags_cumulative_l = plot_cummulatives(u_mags_full, u_full_hops, axs_cum, label =  "Uniform", color = u_color,plot = plot_cums)
    u_mags_mean = np.mean(u_mags_cumulative_l, axis=0)
    
    if got_vals:
        if plot_cums:
            u_distances = [abs(val - exact_mag) for val in u_mags_mean]
            axs_cum[1].plot(u_full_hops[1:], u_distances, color=u_color)


if plot_cums:
    axs_cum[0].set_title('Cumulative magnetization for '+str(n_spins)+" spins")
    axs_cum[0].set_xlabel('Steps')
    axs_cum[0].set_ylabel('Cumulative magnetization')
    axs_cum[0].set_xscale("log")

    fig_cum.delaxes(axs_cum[1])
    axs_cum[0].set_position([0.1, 0.1, 0.8, 0.8])

    #axs_cum[1].set_title('Accuracy of cumulative magnetisation')
    #axs_cum[1].set_xlabel('Steps')
    ##axs_cum[1].set_ylabel("Distance from actual magnetisation")
    #axs_cum[1].set_xscale("log")
    #axs_cum[1].set_yscale("log")
    
#
    if got_vals:
        if do_local:
            axs_cum[0].plot([0,l_full_hops[-1]],[exact_mag,exact_mag],  color = "k", ls = "--", label = "Exact Magnetisation")
        elif do_quantum_Q:
            axs_cum[0].plot([0,Q_full_hops[-1]],[exact_mag,exact_mag],  color = "k", ls = "--", label = "Exact Magnetisation")
        elif do_quantum_q:
            axs_cum[0].plot([0,q_full_hops[-1]],[exact_mag,exact_mag],  color = "k", ls = "--", label = "Exact Magnetisation")

        for i, mm in enumerate(minima_mags):
            color = interpolate_color("springgreen", "darkblue", float((i)/len(lowest_energies)))

    axs_cum[0].legend(loc = "best")
    #fig_cum.legend()
    #plt.xscale("log")
    fig_cum.show()




# %%
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Convergence analysis
print("starting convergence analysis stuff")


if plot_convergence_analysis:






    fig_mconv, axs_mconv = plt.subplots(3,1)
    if do_quantum_Q:
        Q_full_energies = np.array(Q_full_energies)
        Q_mags_full = np.array(Q_mags_full)
        #print("Q_full_energies:", Q_full_energies)
        #print("Q_mags_full:", Q_mags_full)
        Q_r_hat_m = compute_rhat(Q_mags_full, np.arange(0, len(Q_full_hops)))
        axs_mconv[0].plot(Q_full_hops[10:],Q_r_hat_m,label = str(int(m_Q))+" qubits multiple sample", color = Q_color)
        
        Q_r_mcse_m = compute_mcse(Q_mags_full,np.arange(0, len(Q_full_hops)))
        axs_mconv[1].plot(Q_full_hops[10:],Q_r_mcse_m, color = Q_color)
        
        Q_r_ess_m = compute_ess(Q_mags_full,np.arange(0, len(Q_full_hops)))
        axs_mconv[2].plot(Q_full_hops[10:],Q_r_ess_m, color = Q_color)
        #print("Q_r_hat_m:", Q_r_hat_m)
        #print("Q_r_mcse_m:", Q_r_mcse_m)
        #print("Q_r_ess_m:", Q_r_ess_m)
        
    if do_quantum_q:
        q_full_energies = np.array(q_full_energies)
        q_mags_full = np.array(q_mags_full)
        
        q_r_hat_m = compute_rhat(q_mags_full, np.arange(0, len(q_full_hops)))
        axs_mconv[0].plot(q_full_hops[10:],q_r_hat_m,label = str(int(m_q))+" qubits multiple sample", color = q_color)
        
        q_r_mcse_m = compute_mcse(q_mags_full,np.arange(0, len(q_full_hops)))
        axs_mconv[1].plot(q_full_hops[10:],q_r_mcse_m, color = q_color)
        
        q_r_ess_m = compute_ess(q_mags_full,np.arange(0, len(q_full_hops)))
        axs_mconv[2].plot(q_full_hops[10:],q_r_ess_m, color = q_color)
        
        
    if do_quantum_q_2:
        q_2_full_energies = np.array(q_2_full_energies)
        q_2_mags_full = np.array(q_2_mags_full)
        
        q_2_r_hat_m = compute_rhat(q_2_mags_full, np.arange(0, len(q_2_full_hops)))
        axs_mconv[0].plot(q_2_full_hops[10:],q_2_r_hat_m,label = str(int(m_q_2))+" qubits multiple sample", color = q_2_color)
        
        q_2_r_mcse_m = compute_mcse(q_2_mags_full,np.arange(0, len(q_2_full_hops)))
        axs_mconv[1].plot(q_2_full_hops[10:],q_2_r_mcse_m, color = q_2_color)
        
        q_2_r_ess_m = compute_ess(q_2_mags_full,np.arange(0, len(q_2_full_hops)))
        axs_mconv[2].plot(q_2_full_hops[10:],q_2_r_ess_m, color = q_2_color)
        
        
        
    if do_quantum_nms:
        nms_full_energies = np.array(nms_full_energies)
        nms_mags_full = np.array(nms_mags_full)
        
        nms_r_hat_m = compute_rhat(nms_mags_full, np.arange(0, len(nms_full_hops)))
        axs_mconv[0].plot(nms_full_hops[10:],nms_r_hat_m,label = str(int(m_q_nms))+" qubits non-multiple sample", color = nms_color)
        
        nms_r_mcse_m = compute_mcse(nms_mags_full,np.arange(0, len(nms_full_hops)))
        axs_mconv[1].plot(nms_full_hops[10:],nms_r_mcse_m, color = nms_color)
        
        nms_r_ess_m = compute_ess(nms_mags_full,np.arange(0, len(nms_full_hops)))
        axs_mconv[2].plot(nms_full_hops[10:],nms_r_ess_m, color = nms_color)
        
    if do_uniform:
        u_full_energies = np.array(u_full_energies)
        u_mags_full = np.array(u_mags_full)
        u_r_hat_m = compute_rhat(u_mags_full, np.arange(0, len(u_full_hops)))
        axs_mconv[0].plot(u_full_hops[10:],u_r_hat_m,label = "Uniform", color = u_color)
        
        u_r_mcse_m = compute_mcse(u_mags_full,np.arange(0, len(u_full_hops)))
        axs_mconv[1].plot(u_full_hops[10:],u_r_mcse_m, color = u_color)
        
        u_r_ess_m = compute_ess(u_mags_full,np.arange(0, len(u_full_hops)))
        axs_mconv[2].plot(u_full_hops[10:],u_r_ess_m, color = u_color)
        
    if do_local:
        l_full_energies = np.array(l_full_energies)
        l_mags_full = np.array(l_mags_full)
        
        l_r_hat_m = compute_rhat(l_mags_full, np.arange(0, len(l_full_hops)))
        axs_mconv[0].plot(l_full_hops[10:],l_r_hat_m,label = "Local", color = l_color)
        
        l_r_mcse_m = compute_mcse(l_mags_full,np.arange(0, len(l_full_hops)))
        axs_mconv[1].plot(l_full_hops[10:],l_r_mcse_m, color = l_color)
        
        l_r_ess_m = compute_ess(l_mags_full,np.arange(0, len(l_full_hops)))
        axs_mconv[2].plot(l_full_hops[10:],l_r_ess_m, color = l_color)


    axs_mconv[0].set_title('rhat -1')
    axs_mconv[0].set_ylabel('rhat')
    axs_mconv[0].set_yscale("log")
    axs_mconv[0].set_xscale("log")
    axs_mconv[0].set_ylim(bottom = 0.001)


    axs_mconv[1].set_title('msce')
    axs_mconv[1].set_ylabel('msce')
    axs_mconv[1].set_yscale("log")
    axs_mconv[1].set_xscale("log")
    axs_mconv[1].set_ylim(bottom = 0.001)



    axs_mconv[2].set_yscale("log")
    axs_mconv[2].set_title('ess')
    axs_mconv[2].set_ylabel('ess')
    axs_mconv[2].set_xscale("log")
    axs_mconv[0].legend()
    #plt.tight_layout()

    fig_mconv.supxlabel('Hops')
    fig_mconv.suptitle("Convergence tests, magnetisation "+str(n_spins)+" spins, T = "+str(temp))
    fig_mconv.show()




    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!f
    #energy convergence

    fig_econv, axs_econv = plt.subplots(3,1)

    if do_quantum_Q:
        Q_r_hat = compute_rhat(Q_full_energies,np.arange(0, len(Q_full_hops),1))
        axs_econv[0].plot(Q_full_hops[10:],Q_r_hat,label = str(int(m_Q))+" qubits multiple sample", color = Q_color)
        
        Q_r_mcse = compute_mcse(Q_full_energies,np.arange(0, len(Q_full_hops),1))
        axs_econv[1].plot(Q_full_hops[10:],Q_r_mcse,color = Q_color)
        
        Q_r_ess = compute_ess(Q_full_energies,np.arange(0, len(Q_full_hops),1))
        axs_econv[2].plot(Q_full_hops[10:],Q_r_ess,color = Q_color)
        
        
    if do_quantum_q:
        q_r_hat = compute_rhat(q_full_energies,np.arange(0, len(q_full_hops),1))
        axs_econv[0].plot(q_full_hops[10:],q_r_hat,label = str(int(m_q))+" qubits multiple sample", color = q_color)
        
        q_r_mcse = compute_mcse(q_full_energies,np.arange(0, len(q_full_hops),1))
        axs_econv[1].plot(q_full_hops[10:],q_r_mcse,color = q_color)
        
        q_r_ess = compute_ess(q_full_energies,np.arange(0, len(q_full_hops),1))
        axs_econv[2].plot(q_full_hops[10:],q_r_ess,color = q_color)
        
        
    if do_quantum_q_2:
        q_2_r_hat = compute_rhat(q_2_full_energies,np.arange(0, len(q_2_full_hops),1))
        axs_econv[0].plot(q_2_full_hops[10:],q_2_r_hat,label = str(int(m_q_2))+" qubits multiple sample", color = q_2_color)
        
        q_2_r_mcse = compute_mcse(q_2_full_energies,np.arange(0, len(q_2_full_hops),1))
        axs_econv[1].plot(q_2_full_hops[10:],q_2_r_mcse,color = q_2_color)
        
        q_2_r_ess = compute_ess(q_2_full_energies,np.arange(0, len(q_2_full_hops),1))
        axs_econv[2].plot(q_2_full_hops[10:],q_2_r_ess,color = q_2_color)
        
        
    if do_quantum_nms:
        nms_r_hat = compute_rhat(nms_full_energies,np.arange(0, len(nms_full_hops),1))
        axs_econv[0].plot(nms_full_hops[10:],nms_r_hat,label = str(int(m_q_nms))+" qubits non-multiple sample", color = nms_color)
        
        nms_r_mcse = compute_mcse(nms_full_energies,np.arange(0, len(nms_full_hops),1))
        axs_econv[1].plot(nms_full_hops[10:],nms_r_mcse,color = nms_color)
        
        nms_r_ess = compute_ess(nms_full_energies,np.arange(0, len(nms_full_hops),1))
        axs_econv[2].plot(nms_full_hops[10:],nms_r_ess,color = nms_color)

    if do_uniform:
        u_r_hat = compute_rhat(u_full_energies,np.arange(0, len(u_full_hops),1))
        axs_econv[0].plot(u_full_hops[10:],u_r_hat,label = "Uniform", color = u_color)
        
        u_r_mcse = compute_mcse(u_full_energies,np.arange(0, len(u_full_hops),1))
        axs_econv[1].plot(u_full_hops[10:],u_r_mcse,color = u_color)
        
        u_r_ess = compute_ess(u_full_energies,np.arange(0, len(u_full_hops),1))
        axs_econv[2].plot(u_full_hops[10:],u_r_ess,color = u_color)

    if do_local:
        l_r_hat = compute_rhat(l_full_energies,np.arange(0, len(l_full_hops),1))
        axs_econv[0].plot(l_full_hops[10:],l_r_hat,label = "Local", color = l_color)
        
        l_r_mcse = compute_mcse(l_full_energies,np.arange(0, len(l_full_hops),1))
        axs_econv[1].plot(l_full_hops[10:],l_r_mcse,color = l_color)
        
        l_r_ess = compute_ess(l_full_energies,np.arange(0, len(l_full_hops),1))
        axs_econv[2].plot(l_full_hops[10:],l_r_ess,color = l_color)







    axs_econv[0].set_title('rhat -1')
    axs_econv[0].set_ylabel('rhat')
    axs_econv[0].set_yscale("log")
    axs_econv[0].set_xscale("log")
        
    axs_econv[1].set_title('msce')
    axs_econv[1].set_ylabel('msce')
    axs_econv[1].set_yscale("log")
    axs_econv[1].set_xscale("log")

    axs_econv[2].set_yscale("log")
    axs_econv[2].set_title('ess')
    axs_econv[2].set_ylabel('ess')
    axs_econv[2].set_xscale("log")

    axs_econv[0].legend()
    fig_econv.supxlabel('Hops')
    fig_econv.suptitle("Convergence tests of energy, "+str(n_spins)+" spins, T = "+str(temp))
    fig_econv.show()
plt.show()
