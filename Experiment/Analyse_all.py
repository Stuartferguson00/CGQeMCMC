# The above code is importing the `sys` module in Python. The `sys` module provides access to some
# variables used or maintained by the interpreter and to functions that interact with the interpreter.
import sys
import os
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
from cgqemcmc.basic_utils import *
import itertools
from cgqemcmc.energy_models import IsingEnergyFunction, Exact_Sampling
import pickle
import gc
import scipy
import time as tme
import os



def fill_missing_values(x, y, x_min, x_max):
    
    #Function to return a "thin" Markov chain where only accepted updates are stored (for efficient storage) to a more "full" markov chain
    
    x = np.array(x)
    y = np.array(y)
    valid_indices = np.where(x != None)[0]

    # Remove None values from x and y using boolean indexing
    x = x[valid_indices]
    y = y[valid_indices]
    
    if x_min <1:
        x_min = 1


    full_x = np.logspace(max(np.log10(x_min),0), np.log10(x_max + 1), 1000, dtype = int)
    full_x[0] = 0
    full_x = np.unique(full_x)
    full_y = np.zeros_like(full_x, dtype = float)
    count = 0
    current_y = y[count]
    for i, x_ in enumerate(full_x):
        
        if x_ == 0:
            pass
        elif count>=len(x):
            pass
        elif x_ >= x[count]:
            current_y = y[count]
            count += 1
        full_y[i] = current_y
    return full_x, full_y




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
        
        MCMC_chain_obj = results_list[i]
        energies = MCMC_chain_obj.get_current_energy_array()
        positions = MCMC_chain_obj.get_pos_array()
        states = MCMC_chain_obj.get_current_state_array()
        
        
        all_energies.append(energies)
        all_states.append(states)
        hops.append(positions)
        
        #MCMC_chain_obj.get_accepted_energies()
        
        #accepted_energies = MCMC_chain_obj.accepted_energies
        #accepted_positions = MCMC_chain_obj.accepted_positions
        #accepted_states = MCMC_chain_obj.accepted_states
        
        #accepted_energies = np.array(accepted_energies)
        #accepted_states = np.array(accepted_states)
        #accepted_positions = np.array(accepted_positions)

        
        #all_energies.append(accepted_energies)
        #all_states.append(accepted_states)
        #hops.append(accepted_positions)
        
        
    return np.array(all_energies), np.array(all_states), np.array(hops)



def plot_energies(all_energies, hops, ax, label, color):
    # Assumes the data is from sampling a markov chain

    full_energies = []
    for i in range(len(all_energies)):
        full_hops = hops[i]
        full_energies_ = all_energies[i]
        full_energies.append(full_energies_)
        
        if plot_individuals:
            if i >0:
                label_ = None
            else:
                label_ = label +" individual"

            ax.plot(full_hops, full_energies_, label = label_, color = color, alpha = 0.3)
            
            pass
        
    
            
    full_hops= np.array(full_hops)
        
    full_energies = np.array(full_energies)
    ax.plot(full_hops, np.mean(full_energies,axis = 0), label = label, color = color)

    
    full_hops = np.array(full_hops)
    full_energies = np.array(full_energies)

    
    return full_hops, full_energies




def get_results_dir(n_spins,results_dir, token):


    #change model file names for easy file organisation
    str_nspins = str(n_spins).zfill(3)
    results_dir = results_dir + str_nspins + '_'+ token + '.obj'


    return results_dir


    



























n_spins = 16 # size of system
temp = float(1) # temperature of system to analyse


plot_first = True #whether to plot energy graph or not
plot_individuals = False #whether to plot individual MC chains in energy plot
    


do_local = True
l_sampled = True
sampling_interval_l = 10
l_color = "green"

do_uniform = True
u_sampled = True
sampling_interval_u = 10
u_color = "orange"

#multiple sample
do_Q = True
Q_sampled =   True
sampling_interval_Q = 10
Q_color = "red"
m_q = np.sqrt(n_spins) # number of groups is assumed to be sqrt(n)
m_q_str = "000" if m_q/n_spins == 1 else f"{m_q/n_spins * 1000:03.0f}"

#no coarse graining 
do_Q_2 = True
Q_2_sampled =   True
sampling_interval_Q_2 = 10
Q_2_color = "blue"
m_q_2 = n_spins
m_q_2_str = "000" if m_q_2/n_spins == 1 else f"{m_q_2/n_spins * 1000:03.0f}"

#no multiple sample
do_Q_3 = False
Q_3_sampled =   True
sampling_interval_Q_3 = 10
Q_3_color = "lightblue"
m_q_3 = np.sqrt(n_spins)
m_q_3_str = "000" if m_q_3/n_spins == 1 else f"{m_q_3/n_spins * 1000:03.0f}"
















t_str = str(temp)
t_str = t_str.replace(".", "_")
if len(t_str)>6:
    t_str = t_str[:6]
    
str_nspins = str(n_spins).zfill(3)

home = os.path.dirname(os.path.abspath(__file__))


model_dir = home+'/models/' + str_nspins + '.obj'
results_dir = home+'/results/'+t_str+"/"






fileObj = open(model_dir, 'rb')
model_list = pickle.load(fileObj)
fileObj.close()

if len(model_list) == 1:
    m_l = []
    for i in range(500):
        m_l.append(model_list[0])
    model_list = m_l
    

if n_spins < 20:


    m = model_list[0]
    ES = Exact_Sampling(m, 1/temp)
    boltz_dict = ES.get_boltzmann_distribution(1/temp,sorted = True, save_distribution=True)
    min_s_bitstrings = np.array(list(boltz_dict.keys()))[:10]
    min_s_mu = np.array(list(boltz_dict.values()))[:10]
    lowest_energies, degeneracy = m.get_lowest_energies(10)
    lowest_energies_2 = [m.get_energy(s) for s in min_s_bitstrings]
    lowest_energies = np.array(lowest_energies)
    highest_prob_states = min_s_mu*100
    
    got_vals = True
    

    
else:
    got_vals = False
            

            
if  got_vals:
    avg_energy = np.sum(lowest_energies*highest_prob_states/100)

    print(" ")
    print("The following is the sum of probabilities, if it is much lower than 100, avg_energy will be innacurate.")
    print("In this sense, the average energy and magnetisation calculated here is only accurate for lower temperatures")
    print(np.sum(highest_prob_states))
    
    lowest_energy = np.min(lowest_energies)



m = model_list[0]







fig, ax = plt.subplots(1,1)#,figsize=(8, 8))


if do_local:
    l_results_dir = get_results_dir(n_spins,results_dir, "local")
    l_all_energies, l_all_states, l_hops = unpickle(l_results_dir)
    #l_all_energies, l_all_states, l_hops = thin_stuff([l_all_energies, l_all_states, l_hops],thin_l) 
    
    
    
    l_hops = np.arange(0, len(l_hops[0]) * sampling_interval_l, sampling_interval_l)
    l_hops = np.tile(l_hops, (len(l_hops), 1))
    l_full_hops, l_full_energies = plot_energies(l_all_energies, l_hops, ax, "local", l_color)
    
if do_uniform:

    u_results_dir = get_results_dir(n_spins, results_dir, "uniform")
    u_all_energies, u_all_states, u_hops = unpickle(u_results_dir)
    #u_all_energies, u_all_states, u_hops = thin_stuff([u_all_energies, u_all_states, u_hops], thin_u)
    
    u_hops = np.arange(0, len(u_hops[0]) * sampling_interval_u, sampling_interval_u)
    u_hops = np.tile(u_hops, (len(u_hops), 1))
    u_full_hops, u_full_energies = plot_energies(u_all_energies, u_hops, ax, "uniform", u_color)
    
if do_Q:
    Q_results_dir = get_results_dir(n_spins, results_dir, "q_mult_samp_"+m_q_str)
    Q_all_energies, Q_all_states, Q_hops = unpickle(Q_results_dir)
    #Q_all_energies, Q_all_states, Q_hops = thin_stuff([Q_all_energies, Q_all_states, Q_hops], thin_Q)
    
    Q_hops = np.arange(0, len(Q_hops[0]) * sampling_interval_Q, sampling_interval_Q)
    Q_hops = np.tile(Q_hops, (len(Q_hops), 1))
    Q_full_hops, Q_full_energies = plot_energies(Q_all_energies, Q_hops, ax, "q_mult_samp_"+m_q_str, Q_color)
    
if do_Q_2:
    Q_2_results_dir = get_results_dir(n_spins, results_dir, "q_full_" + m_q_2_str)
    Q_2_all_energies, Q_2_all_states, Q_2_hops = unpickle(Q_2_results_dir)
    # Q_2_all_energies, Q_2_all_states, Q_2_hops = thin_stuff([Q_2_all_energies, Q_2_all_states, Q_2_hops], thin_Q_2)
    
    Q_2_hops = np.arange(0, len(Q_2_hops[0]) * sampling_interval_Q_2, sampling_interval_Q_2)
    Q_2_hops = np.tile(Q_2_hops, (len(Q_2_hops), 1))
    Q_2_full_hops, Q_2_full_energies = plot_energies(Q_2_all_energies, Q_2_hops, ax, "q_full_" + m_q_2_str, Q_2_color)
    
if do_Q_3:
    Q_3_results_dir = get_results_dir(n_spins, results_dir, "q_single_samp_" + m_q_3_str)
    Q_3_all_energies, Q_3_all_states, Q_3_hops = unpickle(Q_3_results_dir)
    # Q_3_all_energies, Q_3_all_states, Q_3_hops = thin_stuff([Q_3_all_energies, Q_3_all_states, Q_3_hops], thin_Q_3)
    
    Q_3_hops = np.arange(0, len(Q_3_hops[0]) * sampling_interval_Q_3, sampling_interval_Q_3)
    Q_3_hops = np.tile(Q_3_hops, (len(Q_3_hops), 1))
    Q_3_full_hops, Q_3_full_energies = plot_energies(Q_3_all_energies, Q_3_hops, ax, "q_full_" + m_q_3_str, Q_3_color)






print(l_all_states[:,0])
print(u_all_states[:,0])
print(Q_all_states[:,0])



if  got_vals:
    ax.plot([0,np.max(u_hops[0])+np.max(u_hops[0])/50],[avg_energy,avg_energy], color = "k", label = "Exact average energy")

fig.suptitle("Thermalisation of MCMC for "+str(n_spins)+" spins and T = "+str(temp))
fig.supxlabel("Steps")
fig.supylabel("Energy")
ax.set_xscale("log")
ax.legend()
plt.show()












"""






#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#Uniform
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



try:
    gc.disable()
    start = tme.time()

    
    fileObj = open(results_dir, 'rb')
    results_list = pickle.load(fileObj)
    fileObj.close()

    end = tme.time()
    gc.enable()
    
    
    
except:
    print("Uniform has not been done yet!")
    print(results_dir)
    quit()
    
    
u_all_energies_l = []
u_all_states_l = []
u_hops = []
for i in range(len(results_list)):

    accepted_energies, accepted_positions, accepted_states = results_list[i]
    u_all_energies_l.append(accepted_energies)
    u_hops.append(accepted_positions)
    u_all_states_l.append(accepted_states)



fig, ax = plt.subplots(1,2)#,figsize=(8, 8))


    
u_full_energies = []
for i in range(len(u_hops)):
    u_full_hops, u_full_energies_ = fill_missing_values(u_hops[i], u_all_energies_l[i], 1, 10**np.ceil(np.log10(np.max(np.array(u_hops[0])[1:]))))
    u_full_energies.append(u_full_energies_)
    
    if plot_individuals:
        ax[0].plot(u_full_hops, u_full_energies_, label = "uniform", color = "orange", alpha = 0.3)
        ax[1].plot(u_full_hops, u_full_energies_, label = "uniform", color = "orange", alpha = 0.3)
        pass
u_full_hops= np.array(u_full_hops)
u_full_energies = np.array(u_full_energies)


ax[0].plot(u_full_hops, np.mean(u_full_energies,axis = 0), label = "Uniform", color = "orange")
ax[1].plot(u_full_hops, np.mean(u_full_energies,axis = 0), label = "Uniform", color = "orange")


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

l_results_dir[-14] = "l"
    
results_dir = ''.join(l_results_dir)




try:
    gc.disable()
    fileObj = open(results_dir, 'rb')
    results_list = pickle.load(fileObj)
    fileObj.close()
    gc.enable()
    
except:
    print("Local has not been done yet!")
    print(results_dir)
    quit()



l_all_energies_l = []
l_all_states_l = []
l_hops = []
for i in range(len(results_list)):
    #u_all_energies_l.append(results_list[i].get_current_energy_array())
    #u_all_states_l.append(results_list[i].get_list_markov_chain())
    
    accepted_energies, accepted_positions, accepted_states = results_list[i]
    l_all_energies_l.append(accepted_energies)
    l_hops.append(accepted_positions)
    l_all_states_l.append(accepted_states)



#ax.plot(l_hops, np.mean(np.array(l_all_energies_l),axis = 0), label = "local", color = "green")


l_full_energies = []
#ax.plot(q_hops, np.mean(np.array(Q_all_energies_l),axis = 0),color=color, alpha = 0.7,label = str(m_q)+" qubits")
for i in range(len(l_hops)):
    l_full_hops, l_full_energies_ = fill_missing_values(l_hops[i], l_all_energies_l[i], 1, 10**np.ceil(np.log10(np.max(np.array([item for sublist in l_hops for item in sublist])[1:]))))
    l_full_energies.append(l_full_energies_)
    if plot_individuals:
        ax[0].plot(l_full_hops, l_full_energies_, label = "local", color = "green", alpha = 0.3)
        ax[1].plot(l_full_hops, l_full_energies_, label = "local", color = "green", alpha = 0.3)
        pass
l_full_hops= np.array(l_full_hops)
l_full_energies = np.array(l_full_energies)
ax[0].plot(l_full_hops, np.mean(l_full_energies,axis = 0), label = "Local", color = "green")
ax[1].plot(l_full_hops, np.mean(l_full_energies,axis = 0), label = "Local", color = "green")




















#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#quantum
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


l_results_dir = list(Q_results_dir)
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


Q_results_dir = ''.join(l_results_dir)







    
l_results_dir = list(Q_results_dir)
l_results_dir[-7] = str(m_q/n_spins)[2]
try:
    l_results_dir[-6] = str(m_q/n_spins)[3]
except:
    l_results_dir[-6] = str(0)
try:
    l_results_dir[-5] = str(m_q/n_spins)[4]
except:
    l_results_dir[-5] = str(0)

l_results_dir[-14] = "r"
l_results_dir[-13] = "l"
    
Q_results_dir = ''.join(l_results_dir)






try:
    gc.disable()
    print(Q_results_dir)
    fileObj = open(Q_results_dir, 'rb')
    results_list = pickle.load(fileObj)
    fileObj.close()
    
    gc.enable()

    Q_all_energies_l = []
    Q_all_states_l = []
    q_hops = []
    
    for i in range(len(results_list)):
        accepted_energies, accepted_positions, accepted_states = results_list[i]
        Q_all_energies_l.append(accepted_energies)
        q_hops.append(accepted_positions)
        Q_all_states_l.append(accepted_states)



    
    
except:
    print("couldnt unpickle quantum")
    print(results_dir)
    Q_all_energies_l = []
    Q_all_states_l = []
    q_hops = []
    Q_all_states_l.append(list([]))
    Q_all_energies_l.append(list([]))
    q_hops.append(list([]))
    pass


color = "red"
Q_full_energies = []
for i in range(len(Q_all_energies_l)):
    Q_full_hops, Q_full_energies_ = fill_missing_values(q_hops[i], Q_all_energies_l[i], 1, 10**np.ceil(np.log10(np.max(np.array(q_hops[0])[1:]))))
    Q_full_energies.append(Q_full_energies_)
    if plot_individuals:
        ax[1].plot(Q_full_hops, Q_full_energies_, label = str(int(m_q))+" qubits multiple sample", color = color, alpha = 0.3)
        ax[0].plot(Q_full_hops, Q_full_energies_, label = str(int(m_q))+" qubits multiple sample", color = color, alpha = 0.3)
        pass
        
Q_full_hops= np.array(Q_full_hops)
Q_full_energies = np.array(Q_full_energies)
ax[0].plot(Q_full_hops, np.mean(Q_full_energies,axis = 0), label = str(int(m_q))+" qubits multiple sample", color = color)
ax[1].plot(Q_full_hops, np.mean(Q_full_energies,axis = 0), label = str(int(m_q))+" qubits multiple sample", color = color)



ax[1].set_xscale("log")
ax[0].set_xscale("log")









if n_spins < 20 or got_vals:
    for i, le in enumerate(lowest_energies):
        color = interpolate_color("springgreen", "darkblue", float((i)/len(lowest_energies)))
        if i ==0:
            ax[1].plot([0,np.max(u_hops[0])+np.max(u_hops[0])/50],[le,le], color = color, ls = "--", alpha = 0.3, label = "Lowest energy levels")#, label = str(i)+" excited_sate"
            ax[0].plot([0,np.max(u_hops[0])+np.max(u_hops[0])/50],[le,le], color = color, ls = "--", alpha = 0.3, label = "Lowest energy levels")#, label = str(i)+" excited_sate"

        else:
            ax[1].plot([0,np.max(u_hops[0])+np.max(u_hops[0])/50],[le,le],  color = color, ls = "--", alpha = 0.3)#label = str(i)+" excited_sate",
            ax[0].plot([0,np.max(u_hops[0])+np.max(u_hops[0])/50],[le,le], color = color, ls = "--", alpha = 0.3)#, label = str(i)+" excited_sate"
        

    zoom_y_min, zoom_y_max = np.min(lowest_energies)-1, np.max(lowest_energies)-np.max(lowest_energies)*0.1
    zoom_x_min, zoom_x_max = 1, np.max(list(itertools.chain.from_iterable(u_hops)))


    ax[1].set_xlim(zoom_x_min, zoom_x_max)
    ax[1].set_ylim(zoom_y_min, zoom_y_max)

    ax[0].plot([0,zoom_x_max],[avg_energy,avg_energy], color = "k", label = "Exact average energy")
    ax[1].plot([0,zoom_x_max],[avg_energy,avg_energy], color = "k", label = "Exact average energy")



handles, labels = fig.gca().get_legend_handles_labels()
unique_labels = []
unique_handles = []
for i, label in enumerate(labels):
    if label not in unique_labels:
        unique_labels.append(label)
        unique_handles.append(handles[i])
        

ax[0].legend(unique_handles, unique_labels)

if plot_first:

    fig.suptitle("Thermalisation of MCMC for "+str(n_spins)+" spins")
    fig.supxlabel("Steps")
    fig.supylabel("Energy")
    ax[0].set_title("Full thermalisation")
    ax[1].set_title("Low energy region")
    plt.show()
else:
    plt.close()








#get "raw mags"
Q_mags_l = []
for i in range(len(Q_all_energies_l)):
    Q_mags = []
    for k in range(len(q_hops[i])):
        Q_mags.append(magnetization_of_state(Q_all_states_l[i][k]))
    Q_mags_l.append(Q_mags)

#fill out
q_mags_full = []
q_full_hops_l = []
for i in range(len(Q_mags_l)):
    _, q_mags_full_ = fill_missing_values(q_hops[i], Q_mags_l[i], Q_full_hops[0], Q_full_hops[-1])
    q_mags_full.append(q_mags_full_)
q_mags_full = np.array(q_mags_full)


#get unique counts
q_unique_dict = {}
q_has_lowest = 0
for i in range(len(q_mags_full)):
    unique, count = np.unique(Q_full_energies[i], return_counts = True)
    #print(unique, count)

    if lowest_energy in unique:
        q_has_lowest+=1
    
    for k, u in enumerate(unique):
        try:
            q_unique_dict[u] += float(count[k])
        except:
            q_unique_dict[u] = float(count[k])
            
            





u_mags_l = []
for i in range(len(u_all_energies_l)):
    u_mags = []
    for j in range(len(u_hops[i])):
        u_mags.append(magnetization_of_state(u_all_states_l[i][j]))
    u_mags_l.append(u_mags)
    
    
u_has_lowest = 0
u_mags_full = []
u_full_hops_l = []
for i in range(len(u_mags_l)):
    _, u_mags_full_ = fill_missing_values(u_hops[i], u_mags_l[i], u_full_hops[0], u_full_hops[-1])
    u_mags_full.append(u_mags_full_)
u_mags_full = np.array(u_mags_full)

    
u_unique_dict = {}
for i in range(len(u_full_energies)):
    unique, count = np.unique(u_full_energies[i], return_counts = True)
    
    if lowest_energy in unique:
        u_has_lowest+=1
    
    for k, u in enumerate(unique):
        try:
            u_unique_dict[u] += float(count[k])
        except:
            u_unique_dict[u] = float(count[k])
    












l_mags_l = []
for i in range(len(l_all_energies_l)):
    l_mags = []
    for j in range(len(l_hops[i])):
        l_mags.append(magnetization_of_state(l_all_states_l[i][j]))
    l_mags_l.append(l_mags)
    



l_mags_full = []
l_full_hops_l = []
for i in range(len(l_mags_l)):
    _, l_mags_full_ = fill_missing_values(l_hops[i], l_mags_l[i], l_full_hops[0], l_full_hops[-1])
    l_mags_full.append(l_mags_full_)
l_mags_full = np.array(l_mags_full)


l_has_lowest = 0

l_unique_dict = {}
for i in range(len(l_full_energies)):
    unique, count = np.unique(l_full_energies[i], return_counts = True)
    
    if lowest_energy in unique:
        l_has_lowest+=1
    
    for k, u in enumerate(unique):
        try:
            l_unique_dict[u] += float(count[k])
        except:
            l_unique_dict[u] = float(count[k])
        


    





# Assuming you have defined the necessary variables and lists before this section
# Calculate the logspace for hops

# PLOT CUMULATIVES
print(" ")
print("PLOTTING CUMULATIVES")




def plot_cummulatives(mags_l, _hops,  axs, label =  "Uniform", color = "orange"):
    
    mags_cumulative_l = []
    for j, mags in enumerate(mags_l):
        mags_cumulative = []
        for i in range(len(_hops)):
            if i ==0:
                mags_cumulative.append(mags[i])
            else:
                mags_cumulative.append(np.sum(mags[:i]) / (len(mags[:i])))
            

            
        mags_cumulative_l.append(mags_cumulative)
        #axs[0].plot(_hops, mags_cumulative, color=color, alpha=0.1)
        

    yerrs = scipy.stats.sem(mags_cumulative_l, axis=0)
    mean = np.mean(mags_cumulative_l, axis=0)
    axs[0].plot(_hops, mean, label=label, color=color, alpha=1)
    

    axs[0].fill_between(_hops, mean - yerrs, mean + yerrs, alpha=0.2, color=color)
    
    mags_cumulative_l = np.array(mags_cumulative_l)
    return mags_cumulative_l
    


# Plotting subplot 1: u_mags_cumulative vs. hops
fig, axs = plt.subplots(2, 1)

u_mags_cumulative_l = plot_cummulatives(u_mags_full, u_full_hops, axs, label =  "Uniform", color = "orange")
l_mags_cumulative_l = plot_cummulatives(l_mags_full, l_full_hops, axs, label =  "Local", color = "green")
q_mags_cumulative_l = plot_cummulatives(q_mags_full, Q_full_hops, axs, label =  str(int(m_q))+" qubits multiple sample", color = "red")

l_mags_mean = np.mean(l_mags_cumulative_l, axis=0)
u_mags_mean = np.mean(u_mags_cumulative_l, axis=0)
q_mags_mean = np.mean(q_mags_cumulative_l, axis=0)




axs[0].set_title('Cumulative magnetization for '+str(n_spins)+" spins")
axs[0].set_xlabel('Steps')
axs[0].set_ylabel('Cumulative magnetization')
axs[0].set_xscale("log")



final_value = np.mean(l_mags_cumulative_l[:,-1])
u_distances = [abs(val - exact_mag) for val in u_mags_mean]  # Absolute difference from the final value
l_distances = [abs(val - exact_mag) for val in l_mags_mean]  # Absolute difference from the final value
q_distances = [abs(val - exact_mag) for val in q_mags_mean]  # Absolute difference from the final value

axs[1].plot(u_full_hops, u_distances, color="orange")
axs[1].plot(l_full_hops, l_distances, color="green")
axs[1].plot(Q_full_hops, q_distances, color="red")



axs[1].set_title('Accuracy of cumulative magnetisation')
axs[1].set_xlabel('Steps')
axs[1].set_ylabel("Distance from actual magnetisation")
axs[1].set_xscale("log")
axs[1].set_yscale("log")
# Adjust layout to prevent overlap
plt.tight_layout()
axs[0].legend(loc = "best")

# Show plot
if n_spins < 20 or got_vals:
    axs[0].plot([0,l_full_hops[-1]],[exact_mag,exact_mag],  color = "k", ls = "--", label = "Exact Magnetisation")

if n_spins < 20 or got_vals:
    for i, mm in enumerate(minima_mags):
        color = interpolate_color("springgreen", "darkblue", float((i)/len(lowest_energies)))


plt.legend()
plt.show()
"""