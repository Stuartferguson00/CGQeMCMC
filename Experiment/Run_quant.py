
import sys
import os

# Add the parent home to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cgqemcmc.basic_utils import *
import pickle
import time as tme
import os 
from cgqemcmc.qulacs_CGQeMCMC import MCMC_qulacs
import joblib


def do_quantum_MCMC(i, last_done, multiple_sample, m_q, temp, time, gamma, n_hops, model_list, n_spins, sample_frequency ):
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

        
        MCMC = MCMC_qulacs(m, gamma, time, temp, max_qubits = int(m_q), CG_sample_number = CG_sample_number)

        
        output = MCMC.run(n_hops, initial_state=m.initial_state[last_done +i], sample_frequency=sample_frequency)
        #output = thin_MCMC_chain(output)# just gets accepted energies states etc.
        t = tme.time()-t
        
        print("time taken by thread "+str(i) +"vis "+str(t))
        
        return output



def main(n_spins, temp, reps,n_hops,multiple_sample, m_q,sample_frequency):
    
    if m_q == n_spins:
        proposal = "q_full"
    elif multiple_sample:
        proposal = "q_mult_samp"
    else:
        proposal = "q_single_samp"
    
    
    # get temperature string
    t_str = str(temp)
    t_str = t_str.replace(".", "_")
    if len(t_str)>6:
        t_str = t_str[:6]
    
    print("temperature is: "+ t_str)

    home = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(home+'/results/'+t_str,exist_ok=True)
    
    
    model_dir = home+'/models/'
    Q_results_dir = home+'/results/'+t_str+"/"


    m_q_str = "000" if m_q/n_spins == 1 else f"{m_q/n_spins * 1000:03.0f}"

    #change model file names for easy file organisation
    str_nspins = str(n_spins).zfill(3)
    model_dir = model_dir + str_nspins + '.obj'
    Q_results_dir = Q_results_dir + str_nspins + '_'+ proposal + "_" + m_q_str + '.obj'



    
    gamma = (0.25,0.6)
    time = (2,20)



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


    
    
    #get previous data
    try:
        fileObj = open(Q_results_dir, 'rb')
        result_list = pickle.load(fileObj)
        fileObj.close()
        last_done = len(result_list)

    except:
        #if no previous data, start from start
        last_done = 0
        result_list = []


        
    print("The last model that was done for this experiment:")
    print(last_done)
        
    
    
    

    
    # parallelise and time computation
    t_1  = tme.time()
    result_list_ = joblib.Parallel(n_jobs=reps)(joblib.delayed(do_quantum_MCMC)(i,last_done,multiple_sample,m_q,temp,time,gamma,n_hops,model_list,n_spins,sample_frequency) for i in range(0,reps))
    t_1 = tme.time()-t_1
    
    for r in result_list_:
        result_list.append(r)
    
    print("total time is "+str(t_1))
    fileObj = open(Q_results_dir, 'wb')
    pickle.dump(result_list,fileObj)
    fileObj.close()
    #print(result_list)
        
    
if __name__ == "__main__":
    
 
    # total arguments
    n = len(sys.argv)
    if n != 8:
        print("Invalid arguments. Required arguments are: ")
        print("n_spins(int) temp(float) reps(int) n_hops(int) mult_samp(bool) group_size(int) sample_frequency(int)")
        quit()
    args = []
    for i in range(1, n):
        if i ==1 or i ==3 or i ==4 or i ==6 or i==7:
            args.append(int(sys.argv[i]))
        if i ==2:
            args.append(float(sys.argv[i]))
        if i ==5:
            boolean_value = str(sys.argv[i]).lower() == "true"
            args.append(boolean_value)
        
        

    main(args[0],args[1],args[2], args[3], args[4], args[5], args[6])






