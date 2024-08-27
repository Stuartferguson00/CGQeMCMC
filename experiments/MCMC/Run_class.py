
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
from cgqemcmc.classical_mcmc_routines import MCMC_Classical
import joblib

dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(dir))
sys.path.insert(0, root_dir)

def main(n_spins, temp, reps,n_hops,sample_frequency):
    t_str = str(temp)
    t_str = t_str.replace(".", "_")
    if len(t_str)>6:
        t_str = t_str[:6]

    dir = os.path.dirname(os.path.abspath(__file__))
    print("temperature is: "+ t_str)
    os.makedirs(dir+'/Results/'+t_str,exist_ok=True)
    
    
    
    model_dir = dir+'/Models/000.obj'
    results_dir = dir+'/Results/'+t_str+'/oo_000_000.obj'






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
        


    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #Uniform

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
    results_dir = ''.join(l_results_dir)

    l_results_dir = list(results_dir)

    l_results_dir[-14] = "u"
    #l_results_dir[-12] = str(l)[1]
        
    results_dir = ''.join(l_results_dir)



    try:
        fileObj = open(results_dir, 'rb')
        result_list = pickle.load(fileObj)
        fileObj.close()
        last_done = len(result_list)-1

    except:
        last_done = 0
        result_list = []

        




        
    def do_classical_MCMC(i,last_done,temp,n_hops,model_list, sample_frequency, method = "uniform"):
        if last_done +i > len(model_list)-1:
            print("You have run out of models!")
            return
        else:
            t = tme.time()
            m = model_list[last_done+i]
                        
            MCMC = MCMC_Classical(m, temp,method= method)
            
                        
            output = MCMC.run(n_hops, sample_frequency = sample_frequency, initial_state=m.initial_state[last_done +i])
            
            #thin_output = thin_MCMC_chain(output)

            #print("done "+str(i))
            
            energies = output.get_all_energies()
            pos = output.get_pos_array()
            states = output._states
            
            _states = []
            for state in states:
                _states.append(state.bitstring)
            print([energies, pos, _states], flush=True)
            
            t = tme.time()-t
            print("time taken by thread "+str(i) +"vis "+str(t), flush=True)
        
            return [energies, pos, _states]
            
        #return thin_output
    
        
    print(joblib.cpu_count())
    
    t_1  = tme.time()
    result_list = joblib.Parallel(n_jobs=reps)(joblib.delayed(do_classical_MCMC)(i,last_done,temp,n_hops,model_list,sample_frequency, "uniform") for i in tqdm(range(0,reps)))
    t_1 = tme.time()-t_1
    
    print("total time uniform is "+str(t_1))
    
    print(results_dir)
    fileObj = open(results_dir, 'wb')
    pickle.dump(result_list,fileObj)
    fileObj.close()
    
    
    
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #local
    
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
    results_dir = ''.join(l_results_dir)

    l_results_dir = list(results_dir)

    l_results_dir[-14] = "l"
    #l_results_dir[-12] = "l"
        
    results_dir = ''.join(l_results_dir)
    




    try:
        fileObj = open(results_dir, 'rb')
        result_list = pickle.load(fileObj)
        fileObj.close()
        last_done = len(result_list)-1

    except:
        last_done = 0
        result_list = []
    
    
    
    t_1  = tme.time()
    result_list = joblib.Parallel(n_jobs=reps)(joblib.delayed(do_classical_MCMC)(i,last_done,temp,n_hops,model_list, sample_frequency,"local") for i in tqdm(range(0,reps)))
    t_1 = tme.time()-t_1
    
    print("total time local is "+str(t_1))
    print(results_dir)
    fileObj = open(results_dir, 'wb')
    pickle.dump(result_list,fileObj)
    fileObj.close()
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    


        

if __name__ == "__main__":
    
 
    # total arguments
    n = len(sys.argv)
    if n != 6:
        print("Invalid arguments. Required arguments are: ")
        print("n_spins(int) temp(float) reps(int) n_hops(int) sample_frequency(int)")
        quit()
    args = []
    for i in range(1, n):
        if i ==1 or i ==3 or i ==4 or i==5:
            print(sys.argv[i])
            args.append(int(sys.argv[i]))
        if i ==2:
            args.append(float(sys.argv[i]))
        
        
    #the following variables are not changing between each experiment

    

    main(args[0],args[1],args[2], args[3], args[4])


#main(36, float(0.1), 5, 10**(5))







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
from cgqemcmc.classical_mcmc_routines import MCMC_Classical


import joblib


def do_classical_MCMC(i,last_done,temp,n_hops,model_list, method = "uniform"):
    # function to do a single classical MCMC
    t = tme.time()
    if last_done +i > len(model_list)-1:
        print("You have run out of models!")
        exit
    else:
        m = model_list[last_done+i]
                
        MCMC = MCMC_Classical(m, temp,method= method)
        
                
        output = MCMC.run(n_hops)
        
        thin_output = thin_MCMC_chain(output)

        t = tme.time()-t
    
    print("time taken by thread "+str(i) +"vis "+str(t))
        
    return thin_output

def main(n_spins, temp, reps,n_hops):
    
    
    #find temp string
    t_str = str(temp)
    t_str = t_str.replace(".", "_")
    if len(t_str)>6:
        t_str = t_str[:6]

    dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(dir+'/results/'+t_str,exist_ok=True)
    
    model_dir = dir+'/models_001/000.obj'

    results_dir = dir+'/results/'+t_str+'/oo_000_000.obj'


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
        


    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Run uniform MCMC

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
    results_dir = ''.join(l_results_dir)

    l_results_dir = list(results_dir)

    l_results_dir[-14] = "u"
        
    results_dir = ''.join(l_results_dir)



    #get previous data
    try:
        fileObj = open(results_dir, 'rb')
        result_list = pickle.load(fileObj)
        fileObj.close()
        last_done = len(result_list)-1

    except:
        #if no previous data, start from start
        last_done = 0
        result_list = []

        
    t_1  = tme.time()
    result_list = joblib.Parallel(n_jobs=reps)(joblib.delayed(do_classical_MCMC)(i,last_done,temp,n_hops,model_list, "uniform") for i in tqdm(range(0,reps)))
    t_1 = tme.time()-t_1
    
    print("total time taken by uniform is "+str(t_1))
    
    fileObj = open(results_dir, 'wb')
    pickle.dump(result_list,fileObj)
    fileObj.close()
    
    
    
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #Run local MCMC 
    
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
    results_dir = ''.join(l_results_dir)

    l_results_dir = list(results_dir)

    l_results_dir[-14] = "l"
        
    results_dir = ''.join(l_results_dir)
    




    try:
        fileObj = open(results_dir, 'rb')
        result_list = pickle.load(fileObj)
        fileObj.close()
        last_done = len(result_list)-1

    except:
        last_done = 0
        result_list = []
    
    
    
    t_1  = tme.time()
    result_list = joblib.Parallel(n_jobs=reps)(joblib.delayed(do_classical_MCMC)(i,last_done,temp,n_hops,model_list, "local") for i in tqdm(range(0,reps)))
    t_1 = tme.time()-t_1
    
    print("total time taken by local is "+str(t_1))
    fileObj = open(results_dir, 'wb')
    pickle.dump(result_list,fileObj)
    fileObj.close()
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    


        

if __name__ == "__main__":
    
 
    # total arguments
    n = len(sys.argv)
    if n != 5:
        print("Invalid arguments. Required arguments are: ")
        print("n_spins(int) temp reps(int) n_hops")
        quit()
    args = []
    for i in range(1, n):
        if i ==1 or i ==3 or i ==4:
            print(sys.argv[i])
            args.append(int(sys.argv[i]))
        if i ==2:
            args.append(float(sys.argv[i]))


    main(args[0],args[1],args[2], args[3])




"""