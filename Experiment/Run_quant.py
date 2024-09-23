
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cgqemcmc.basic_utils import *
import pickle
import time as tme
import os 
from cgqemcmc.qulacs_CGQeMCMC import MCMC_qulacs
import joblib


def do_quantum_MCMC(i, last_done, multiple_sample, m_q, temp, time, gamma, n_hops, model_list, n_spins ):
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

        
        output = MCMC.run(n_hops)

        thin_output = thin_MCMC_chain(output)
        t = tme.time()-t
        
        print("time taken by thread "+str(i) +"vis "+str(t))
        
        return thin_output



def main(n_spins, temp, reps,n_hops,multiple_sample, m_q):
    
    
    # get temperature string
    t_str = str(temp)
    t_str = t_str.replace(".", "_")
    if len(t_str)>6:
        t_str = t_str[:6]
    
    print("temperature is: "+ t_str)

    dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(dir+'/results/'+t_str,exist_ok=True)
    
    
    model_dir = dir+'/models/000.obj'




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

    l_results_dir[-14] = "l"
        
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
    result_list = joblib.Parallel(n_jobs=reps)(joblib.delayed(do_quantum_MCMC)(i,last_done,multiple_sample,m_q,temp,time,gamma,n_hops,model_list,n_spins) for i in range(0,reps))
    t_1 = tme.time()-t_1
    
    print("total time is "+str(t_1))
    fileObj = open(Q_results_dir, 'wb')
    pickle.dump(result_list,fileObj)
    fileObj.close()
    #print(result_list)
        
    
if __name__ == "__main__":
    
 
    # total arguments
    n = len(sys.argv)
    if n != 7:
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
        
        

    

    main(args[0],args[1],args[2], args[3], args[4], args[5])






