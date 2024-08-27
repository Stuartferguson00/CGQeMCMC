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
    os.makedirs(dir+'/Results/'+t_str,exist_ok=True)
    model_dir = dir+'/Models/000.obj'

    print("multiple_sample")
    print(multiple_sample)
    print(not multiple_sample)
    if not multiple_sample:
        Q_results_dir = dir+'/Results/'+t_str+'/non_ms_oo_000_000.obj'
    else:
        Q_results_dir = dir+'/Results/'+t_str+'/oo_000_000.obj'
    
    print("Q_results_dir")
    print(Q_results_dir)
    
    build_on_other_saves = True
    
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
    if noise_model is None:
        l_results_dir[-14] = "r"
        l_results_dir[-13] = "l"
    elif noise_model == "Depolarizing":
        l_results_dir[-14] = "d"
        l_results_dir[-13] = "p"
    else:
        print("noise model is wrong")
        
        
    Q_results_dir = ''.join(l_results_dir)
    print("Q_results_dir")
    print(Q_results_dir)
    
    if build_on_other_saves:
        try:
            fileObj = open(Q_results_dir, 'rb')
            result_list = pickle.load(fileObj)
            fileObj.close()
            last_done = len(result_list)-1

        except:
            last_done = 0
            result_list = []
    else:
        last_done = 0
        result_list = []
    
        
        
    print("The last model that was done for this experiment:", flush=True)
    print(last_done, flush=True)
        
    
    
    
    def do_quantum_MCMC(i,last_done,multiple_sample,m_q,temp,time,gamma,n_hops,model_list,sample_frequency, GPU_avail, noise_model, noise_prob_one_qubit, noise_prob_two_qubit):
        if GPU_avail:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
        
        t = tme.time()
        if last_done +i > len(model_list)-1:
            print("You have run out of models!", flush=True)
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

            
            MCMC = MCMC_qulacs(m, gamma, time, temp, max_qubits = int(m_q), CG_sample_number = CG_sample_number, noise_model=noise_model, noise_prob_one_qubit = noise_prob_one_qubit, noise_prob_two_qubit = noise_prob_two_qubit)

            
            output = MCMC.run(n_hops, sample_frequency  =sample_frequency, initial_state=m.initial_state[last_done +i] )

            #thin_output = thin_MCMC_chain(output)
            t = tme.time()-t
            print("time taken by thread "+str(i) +"vis "+str(t), flush=True)
            
            
            #return thin_output
            #return output
            
            energies = output.get_all_energies()
            pos = output.get_pos_array()
            states = output._states
            
            _states = []
            for state in states:
                _states.append(state.bitstring)
            print([energies, pos, _states], flush=True)
            return [energies, pos, _states]
            
            

    
    print("number of cpus: "+str(joblib.cpu_count()), flush=True)
    
    t_1  = tme.time()
    result_list = joblib.Parallel(n_jobs=reps)(joblib.delayed(do_quantum_MCMC)(i,last_done,multiple_sample,m_q,temp,time,gamma,n_hops,model_list,sample_frequency, GPU_avail,noise_model, noise_prob_one_qubit, noise_prob_two_qubit) for i in range(0,reps))
    t_1 = tme.time()-t_1
    
    print("total time is "+str(t_1), flush=True)
    fileObj = open(Q_results_dir, 'wb')
    pickle.dump(result_list,fileObj)
    fileObj.close()
    #print(result_list)

        
    
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
        
    #the following variables are not changing between each experiment

    

    #main(args[0],args[1],args[2], args[3], args[4], args[5], args[6])
    
    noise_model = None#"Depolarizing"
    noise_prob = 0#0.1
    
    #noise_model = "Depolarizing"
    #noise_prob = 0.01
    
    main(16,float(0.1),7,1000,10,4,True, noise_model, noise_prob_one_qubit = noise_prob, noise_prob_two_qubit = noise_prob)

#main(36, float(0.1), 5, 10**(3))









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



