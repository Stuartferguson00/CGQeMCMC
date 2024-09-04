#!/usr/bin/env python3
import sys
import os

dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(dir))
sys.path.insert(0, root_dir)

from cgqemcmc.basic_utils import *
import pickle
import time as tme
from cgqemcmc.qulacs_CGQeMCMC import MCMC_qulacs
from cgqemcmc.energy_models import IsingEnergyFunction, Exact_Sampling

import joblib
import torch

dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(dir))


def main(n_spins, temp, reps,n_hops, sample_frequency,m_q, multiple_sample, noise_model = None, noise_prob_one_qubit = 0, noise_prob_two_qubit = 0):
    
    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}", flush=True)
    if num_gpus !=0:
        if reps<num_gpus:
            print("number of GPUs must be => number of reps", flush=True)
            quit()
        GPU_avail = True
    else:
        print("No GPUs available", flush=True)
        GPU_avail = False


    t_str = str(temp)
    t_str = t_str.replace(".", "_")
    if len(t_str)>6:
        t_str = t_str[:6]

    dir = os.path.dirname(os.path.abspath(__file__))
    print("temperature is: "+ t_str, flush=True)
    
    model_dir = dir+'/Models/000.obj'



    

    
    
    gamma = (0.25,0.6)
    time = (2,20)



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



    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #quantum
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        


    
    def do_lots_of_s_to_sprime(s,multiple_sample,m_q,temp,time,gamma,n_hops,model_list,sample_frequency, GPU_avail, noise_model, noise_prob_one_qubit, noise_prob_two_qubit):
        if GPU_avail:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
        
        t = tme.time()
        
        m = model_list[0]

        if multiple_sample:
            if n_spins//m_q < n_spins/m_q:
                CG_sample_number = int(n_spins//m_q+1)
            else:
                CG_sample_number = int(n_spins//m_q)
            
        else:
            CG_sample_number = 1

        
        MCMC = MCMC_qulacs(m, gamma, time, temp, max_qubits = int(m_q), CG_sample_number = CG_sample_number, noise_model=noise_model, noise_prob_one_qubit = noise_prob_one_qubit, noise_prob_two_qubit = noise_prob_two_qubit)

        
        #output = MCMC.run(n_hops, sample_frequency  =sample_frequency, initial_state=m.initial_state[last_done +i] )

        proposed_energies, s_energy = MCMC.run_many_hops_constant_s(n_hops,s)
        

        return proposed_energies, s_energy
            
            

    
    
    
    
    m = model_list[0]
    ES = Exact_Sampling(m, 1/temp)
    boltz_dict = ES.get_boltzmann_distribution(1/temp,sorted = True, save_distribution=True)
    min_s_bitstrings = np.array(list(boltz_dict.keys()))[:30]

    print("min_s_bitstrings")
    print(min_s_bitstrings)
    
    print("number of cpus: "+str(joblib.cpu_count()), flush=True)
    
    #s_list = [np.zeros(n_spins)]
    #for s in s_list:
    #result_list = joblib.Parallel(n_jobs=reps)(joblib.delayed(do_quantum_MCMC)(i,last_done,multiple_sample,m_q,temp,time,gamma,n_hops,model_list,sample_frequency, GPU_avail,noise_model, noise_prob_one_qubit, noise_prob_two_qubit) for i in range(0,reps))
    #s = "".join(str(i) for i in(np.zeros(n_spins)))
    #s = np.zeros(n_spins)
    #print(s)
    
    """for i, s in enumerate(min_s_bitstrings):
        if i//10 == i/10:
            c = interpolate_color("springgreen", "darkblue", float((i)/len(min_s_bitstrings)))

            proposed_energies, s_energy = do_lots_of_s_to_sprime(s,multiple_sample,m_q,temp,time,gamma,n_hops,model_list, sample_frequency, GPU_avail,noise_model, noise_prob_one_qubit, noise_prob_two_qubit)
            unique_values, counts = np.unique(proposed_energies, return_counts=True)
            plt.bar(unique_values, counts, alpha = 0.5, color = c)
            #plt.stairs(counts, bins, color = c, fill = True, alpha = 0.5)
            plt.plot([s_energy,s_energy], [0, max(counts)], color = c, label = str(i)+"th lowest state")"""
        
        
    import scipy.stats as stats

    
    # Function to fit a Poisson distribution
    def poisson_distribution(x, mu):
        poisson = stats.poisson.pmf(x, mu)
        return poisson

    tot_proposed_energies = []
    s_energies = []
    for i, s in enumerate(min_s_bitstrings):
        if i % 10 == 0:
            c = interpolate_color("springgreen", "darkblue", float(i) / len(min_s_bitstrings))

            # Simulate the energy states (assuming this is your function)
            proposed_energies, s_energy = do_lots_of_s_to_sprime(s, multiple_sample, m_q, temp, time, gamma, n_hops, model_list, sample_frequency, GPU_avail, noise_model, noise_prob_one_qubit, noise_prob_two_qubit)
            tot_proposed_energies.append(proposed_energies)
            s_energies.append(s_energy)
    tot_unique_values, tot_counts = np.unique(tot_proposed_energies, return_counts=True)

    for i, s in enumerate(min_s_bitstrings):
        if i % 10 == 0:
            c = interpolate_color("springgreen", "darkblue", float(i) / len(min_s_bitstrings))

            # Simulate the energy states (assuming this is your function)
            proposed_energies = tot_proposed_energies[i // 10]
            s_energy = s_energies[i // 10]

            # Get unique energy levels and their counts
            counts = []
            for u in tot_unique_values:
                indices = np.where(proposed_energies == u)[0]
                counts.append(len(indices))

            unique_index = np.arange(0, len(tot_unique_values))

            popt, _ = curve_fit(poisson_distribution, unique_index, counts, p0=[i])
            
            # Overlay the fitted Poisson distribution
            x = unique_index
            y = poisson_distribution(x, *popt)
            
            plt.plot(x, y, color=c, label=f'{i}th lowest state (mu={popt[0]:.2f})')
            print(counts/np.sum(counts))
            plt.bar(unique_index, counts/np.sum(counts), alpha=0.5, color=c)
            plt.plot([i, i], [0, max(tot_counts)/np.sum(tot_counts)], color=c, linestyle='--', label=f'{i}th lowest state')

    plt.ylim(0, 0.1)
    plt.ylabel("Counts")
    plt.xlabel("Energy")
    plt.legend()
    plt.show()




if __name__ == "__main__":
    def boolean_string(s):
        if s not in {'False', 'True'}:
            raise ValueError('Not a valid boolean string')
        return s == 'True'
 
    # total arguments
    n = len(sys.argv)
    if n != 1:
        print("Invalid arguments. Required arguments are: ")
        print("n_spins(int) temp reps(int) n_hops(int) sample_frequency(int) max_qubits(int), multiple_sample(bool)")
        quit()
    args = []
    for i in range(1, n):
        if i ==1 or i ==3 or i ==4 or i ==5 or i ==6:
            print(sys.argv[i])
            args.append(int(sys.argv[i]))
        elif i ==2:
            args.append(float(sys.argv[i]))
        elif i ==7:
            args.append(boolean_string(sys.argv[i]))
        

    

    #main(args[0],args[1],args[2], args[3], args[4], args[5], args[6])
    
    noise_model = None#"Depolarizing"
    noise_prob = 0#0.1
    
    #noise_model = "Depolarizing"
    #noise_prob = 0.01
    
    main(9,float(0.1),1,1000,10,9,True, noise_model, noise_prob_one_qubit = noise_prob, noise_prob_two_qubit = noise_prob)










"""
#!/usr/bin/env python3
import sys
import os
dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(dir))
sys.path.insert(0, root_dir)
from cgqemcmc.basic_utils import *
import pickle
import time as tme
import os 
from cgqemcmc.qulacs_CGQeMCMC import MCMC_qulacs
import joblib
from scipy.stats import poisson, norm, truncnorm
from scipy.optimize import curve_fit


def do_quantum_MCMC(i, last_done, multiple_sample, m_q, temp, time, gamma, n_hops, model_list, n_spins, noise_model, noise_prob):
    # function to do a single quantum MCMC
    t = tme.time()
    if last_done +i > len(model_list)-1:
        print("You have run out of models!")
        exit
    else:
        m = model_list[last_done+i]

        if multiple_sample:
            if n_spins//m_q < n_spins/m_q:
                CG_sample_number = int(n_spins//m_q+1)
            else:
                CG_sample_number = int(n_spins//m_q)
            
        else:
            CG_sample_number = 1

        
        MCMC = MCMC_qulacs(m, gamma, time, temp, max_qubits = int(m_q), CG_sample_number = CG_sample_number, noise_model=noise_model, noise_prob = noise_prob)

        
        output = MCMC.run(n_hops)

        thin_output = thin_MCMC_chain(output)
        t = tme.time()-t
        
        print("time taken by thread "+str(i) +"vis "+str(t))
        
        return thin_output



def main(n_spins, temp, reps,n_hops,multiple_sample, m_q, noise_model = None, noise_prob = 0):
    
    
    # get temperature string
    t_str = str(temp)
    t_str = t_str.replace(".", "_")
    if len(t_str)>6:
        t_str = t_str[:6]
    
    print("temperature is: "+ t_str)

    dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(dir+'/results/'+t_str,exist_ok=True)
    
    
    model_dir = dir+'/models_001/000.obj'



    Q_results_dir = dir+'/results/'+t_str+'/oo_000_000.obj'
    

    
    gamma = (0.25,0.6)
    time = (2,20)


    #change model file names for easy file organisation
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

    #if model list is 1 long, multiply it to emulate a 500 long list of identical models
    if len(model_list) == 1:
        m_l = []
        for i in range(500):
            m_l.append(model_list[0])
        model_list = m_l
    
        



    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Quantum-enhanced MCMC
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Correctly label the output file
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
    results_dir = ''.join(l_results_dir)

    l_results_dir = list(results_dir)

        
    Q_results_dir = ''.join(l_results_dir)


    # Add fraction corresponding to coarse graining to filename
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
    
    
    
    if noise_model is not None:
        if noise_model == "depolarising":
            l_results_dir[-14] = "d"
            l_results_dir[-13] = "p"
        else:
            print("Incorrect nois emodel string or I havent coded name for this noise model yet")
            quit()
    else:
        l_results_dir[-14] = "r"
        l_results_dir[-13] = "l"
        
    Q_results_dir = ''.join(l_results_dir)

    
    
    #get previous data
    try:
        fileObj = open(Q_results_dir, 'rb')
        result_list = pickle.load(fileObj)
        fileObj.close()
        last_done = len(result_list)-1

    except:
        #if no previous data, start from start
        last_done = 0
        result_list = []

    
        
        
    print("The last model that was done for this experiment:")
    print(last_done)
        
    
    
    

    
    # parallelise and time computation
    t_1  = tme.time()
    result_list = joblib.Parallel(n_jobs=reps)(joblib.delayed(do_quantum_MCMC)(i,last_done,multiple_sample,m_q,temp,time,gamma,n_hops,model_list,n_spins, noise_model, noise_prob) for i in range(0,reps))
    t_1 = tme.time()-t_1
    
    print("total time is "+str(t_1))
    fileObj = open(Q_results_dir, 'wb')
    pickle.dump(result_list,fileObj)
    fileObj.close()
    #print(result_list)
        
    
if __name__ == "__main__":
    
 
    # total arguments
    n = len(sys.argv)
    #if n != 7:
    if n != 1:
        print("Invalid arguments. Required arguments are: ")
        print("n_spins(int) temp(float) reps(int) n_hops(int) mult_samp(bool) group_size(int)")
        quit()
    args = []
    for i in range(1, n):
        if i ==1 or i ==3 or i ==4 or i ==6:
            print(sys.argv[i])
            args.append(int(sys.argv[i]))
        if i ==2:
            args.append(float(sys.argv[i]))
        if i ==5:
            args.append(bool(sys.argv[i]))
        
        

    
    noise_model =   "depolarising"
    noise_prob = 0.1
    main(9, 1.0, 1, 100, False, 3, noise_model, noise_prob = noise_prob)
    #main(args[0],args[1],args[2], args[3], args[4], args[5], noise_model)


"""



