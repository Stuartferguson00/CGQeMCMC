##########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
import itertools
from typing import Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
from .basic_utils import qsm, states, MCMCChain, MCMCState
# from .prob_dist import *
from .energy_models import IsingEnergyFunction, Exact_Sampling
from .classical_mcmc_routines import test_accept, get_random_state
from qulacsvis import circuit_drawer
from scipy.linalg import expm
from qulacs.gate import DenseMatrix
from qulacs.gate import X, Y, Z  , Pauli, Identity, merge
from scipy.sparse.linalg import eigs
import scipy
from itertools import combinations
import random
from .energy_models import IsingEnergyFunction, Exact_Sampling



#from .Circuit_Maker_qulacs import Hamming_Circuit_Maker as Circuit_Maker
from .Circuit_Maker_qulacs import Circuit_Maker




class MCMC_qulacs:
    #class to set up the Quantum-enhanced Markov Chain Monte Carlo,
    #allowing for Coarse Graining 
    #also allows for spectral gap to be calculated
    
    def __init__(self, model, gamma, time, temp, max_qubits = None, CG_sample_number = 1, Q= None, A= None, single_qubit_mixer= True, brute_sample =False, brute_sample_multiplier = 1, naive = False):
        ### single qubit mixer is ignored in circuit evaluations currently
        ### I have added some variables here (such as temperature) that I may want to change between iterations
        # So probably best to add as parameter to "run_MCMC" or "find_spec_gap"
        self.Q = Q
        self.brute_sample = brute_sample
        self.brute_sample_multiplier = brute_sample_multiplier
        self.A =A
        self.model = model
        self.gamma = gamma
        self.time = time
        self.temp = temp
        self.beta = 1/self.temp
        self.n_spins = model.num_spins
        self.delta_time = 0.8
        self.max_qubits = max_qubits
        self.CG_sample_number = CG_sample_number
        self.naive = naive
        
        
        avg_sample_size = self.n_spins / self.CG_sample_number
        #print("avg_sample_size: "+str(avg_sample_size))
        if avg_sample_size == self.max_qubits:
            self.sample_sizes = np.ones(self.CG_sample_number)*self.max_qubits
        elif avg_sample_size <= self.max_qubits:
            self.sample_sizes = np.ones(self.CG_sample_number,dtype = int)*self.max_qubits
            self.sample_sizes[-1] = self.n_spins-np.sum(self.sample_sizes[:-1])

        else:
            self.sample_sizes = [self.max_qubits,]


        
        self.single_qubit_mixer = single_qubit_mixer

        self.S = None

        

        if self.max_qubits==self.model.num_spins or self.max_qubits  is None:
            self.course_graining = False
        else:
            self.course_graining = True


    def find_spec_gap_q(self):

        #check if Q is provided
        #useful for cycling through Temps
        self.model.get_all_energies()
        if self.Q is None:
            self.Q = self.find_prob_matrix()
            
        """ 
        print("S")
        print(self.S)
        print("Q")
        print(self.Q)
        """
        #check if A is provided
        #unsure when this wil be useful
        if self.A is None:
            self.A = self.find_acceptance_matrix()

        P = np.multiply(self.Q, self.A)
        

        #account for rejected swaps to add to s = s' matrix element
        for i in range(P.shape[0]):

            #sum all the rejected ones and add to diagonal elements
            #s = np.sum(abs(Q[i,:]-P[i,:]))-abs(Q[i,i]-P[i,i])
            #P[i,i] +=s
            s = np.sum(P[i, :])

            P[i, i] =1- s


        #normalise by row
        row_sums = P.sum(axis=1)
        P = P / row_sums[:, np.newaxis]

        #fine eigenvalues
        e_vals, e_vecs = scipy.linalg.eig(P)
        e_vals = np.flip(np.sort(abs(e_vals)))
        
        #find spectral gap
        delta = e_vals[1]
        delta = 1 - delta
        
        self.delta = delta
        
        return delta


    def run(self,
        n_hops: int,
        initial_state: Optional[str] = None,
        name:str = "quMCMC",
        verbose:bool = False):
        """
        Last 'return_last_n_states' elements of states so collected (default value=500). one can then deduce the distribution from it!
        """



        if initial_state is None:
            initial_state = MCMCState(get_random_state(self.n_spins), accepted=True)
        
        else:
            initial_state = MCMCState(initial_state, accepted=True)
        
        current_state: MCMCState = initial_state
        energy_s = self.model.get_energy(current_state.bitstring)
        initial_state.energy = energy_s

        if verbose: print("starting with: ", current_state.bitstring, "with energy:", energy_s)




        if type(self.gamma) is float:
            gamma = self.gamma*np.ones(n_hops)
        elif type(self.gamma) is tuple:
            gamma = np.round(np.random.uniform(low= min(self.gamma), high = max(self.gamma),size = n_hops), decimals=6)
        else:
            print("gamma is wrong type")
            
        if type(self.time) is int:
            time = self.time*np.ones(n_hops)
        elif type(self.time) is tuple:
            time = np.random.randint(low= np.min(self.time), high = np.max(self.time),size = n_hops)
        else:
            print("time is wrong type")
        

        mcmc_chain = MCMCChain([current_state], name= name)

        for i in tqdm(range(0, n_hops), desc='runnning quantum MCMC steps . ..', disable= not verbose ):
            s_prime = self.get_s_prime(current_state,gamma[i], time[i])

            energy_sprime = self.model.get_energy(s_prime)
            accepted = test_accept(
                energy_s, energy_sprime, temperature=self.temp
            )
            mcmc_chain.add_state(MCMCState(s_prime, accepted, energy_sprime, pos = i))


            if accepted:
                current_state = mcmc_chain.current_state
                energy_s = self.model.get_energy(current_state.bitstring)

        return mcmc_chain
    
    def get_s_prime(self,current_state, g, t):
        
        if g is None:
            if type(self.gamma) is tuple:
                g = np.round(np.random.uniform(low= min(self.gamma), high = max(self.gamma),size = 1), decimals=6)
            else:
                g = self.gamma
        if t is None:
            if type(self.time) is tuple:
                t = np.random.randint(low= np.min(self.time), high = np.max(self.time),size = 1)
            else:
                t = self.time

        
        if not self.course_graining:
            CM = Circuit_Maker(self.model, g, t, self.single_qubit_mixer)
            try:
                s_prime = CM.get_state_obtained_binary(current_state.bitstring)
            except:
                s_prime = CM.get_state_obtained_binary(current_state) 
        else:
            try:
                s_prime = self.sample_transitions_CG_binary(current_state.bitstring, self.max_qubits, g, t)
            except:
                s_prime = self.sample_transitions_CG_binary(current_state, self.max_qubits, g, t)

        return s_prime

        
    def find_prob_matrix(self):
        
        if self.S is None:
            self.S = [''.join(i) for i in itertools.product('01', repeat=self.n_spins)]
        #If time and gamma are single
        if type(self.time) is int and type(self.gamma) is float:
            
            #print("If time and gamma are single")
            if self.brute_sample:
                
                #If  brute sample
                print("brute sample")
                
                Q = np.zeros((2 ** self.n_spins, 2 ** self.n_spins))
                for m in range(2**self.n_spins*self.brute_sample_multiplier):
                    for i, s in enumerate(self.S):
                        binary = self.get_s_prime(s,self.gamma,self.time)
                        if int(binary,2)<len(Q):
                            Q[i,int(binary,2)] +=1
                        else:
                            print("It happened again!")
                row_sums = Q.sum(axis=1)
                row_sums[row_sums == 0] = 0.0000000001
                Q = Q / row_sums[:, np.newaxis]
                    
                    
                
                
            else:
                if self.max_qubits == None or self.max_qubits == self.n_spins:
                    Q = np.zeros((2 ** self.n_spins, 2 ** self.n_spins))
                    energies = []
                    for i, s in enumerate(self.S):
                        CM = Circuit_Maker(self.model, self.gamma, self.time, self.single_qubit_mixer)
                        probs = CM.get_state_obtained_probs(s)
                        Q[i, :] += probs
                    

                else:
                    #If coarse grained

                    Q = np.zeros((2 ** self.n_spins, 2 ** self.n_spins))
                    all_combs = self.generate_combinations(self.n_spins, self.max_qubits)
                    for k, s in enumerate(self.S):
                        for comb in all_combs:
                            print("sample_transitions_CG_probs_row doesn't really work with input  choices yet.")
                            S_, probs = self.sample_transitions_CG_probs_row(s, self.max_qubits, self.gamma, self.time, comb)
                            
                            S_ = [int(''.join(map(str, row)), 2) for row in S_]
                            for j in range(len(S_)):
                                #print(probs[j])
                                #print(S_[j])
                                Q[k, S_[j]] += probs[j]
                            
                row_sums = Q.sum(axis=1)
                row_sums[row_sums == 0] = 0.0000000001
                Q = Q / row_sums[:, np.newaxis]
            
            
        
        elif type(self.time) is tuple and type(self.gamma) is tuple:
            
            #If time and gamma are ranges
            
            if self.brute_sample:
                
                #If  brute sample
                #ie scaling:    #  linearly with Coarse graining    #  linearly with time range      #linearly with gamma range             
                #n_ty_samples =int(abs(self.n_spins- self.max_qubits+1)*np.max([1,(max(self.time) - min(self.time))])*np.max([1,(max(self.gamma*5) - min(self.gamma*5))]))
                #n_ty_samples =int(np.max([1,(max(self.time) - min(self.time))])*np.max([1,(max(self.gamma*5) - min(self.gamma*5))])*self.brute_sample_multiplier)
                n_ty_samples =int(2**self.n_spins*self.brute_sample_multiplier)
                times = np.random.randint(min(self.time), max(self.time),size= (n_ty_samples))
                print(n_ty_samples)
                gammas = np.random.randint(int(min(self.gamma)*100),int(max(self.gamma)*100),size= (n_ty_samples))/100
                

                Q = np.zeros((2 ** self.n_spins, 2 ** self.n_spins))
                for n in range(n_ty_samples):
                
                    #pick random time and gamma
                    t_ = times[n]
                    g_ = gammas[n]
                    for i, s in enumerate(self.S):
                        binary = self.get_s_prime(s,g_,t_)
                        if int(binary,2)<len(Q):
                            Q[i,int(binary,2)] +=1
                        else:
                            print("It happened again!")
                row_sums = Q.sum(axis=1)
                row_sums[row_sums == 0] = 0.0000000001
                Q = Q / row_sums[:, np.newaxis]
                    
                    
            else:
                if self.max_qubits == None or self.max_qubits == self.n_spins:
                    #print("this method randomly samples from each of the time and gamma lists")
                    #just this value for now... it's a little rough...
                    #n_ty_samples = abs(self.n_spins- self.max_qubits+1)*(max(self.time) - min(self.time))


                    #ie scaling:          # linearly with time range         # linearly with gamma range             
                    n_ty_samples =int(np.max([1,(max(self.time) - min(self.time))])*np.max([1,(max(self.gamma*5) - min(self.gamma*5))])*self.brute_sample_multiplier)
                    
                    times = np.random.randint(min(self.time), max(self.time),size= (n_ty_samples))
                    gammas = np.random.randint(int(min(self.gamma)*100),int(max(self.gamma)*100),size= (n_ty_samples))/100

                    
                    Q_ = np.zeros((2 ** self.n_spins, 2 ** self.n_spins))
                    for i in range(n_ty_samples):
                        #pick random time and gamma
                        t_ = times[i]
                        g_ = gammas[i]
                        
                        #initialise particular circuit base
                        CM = Circuit_Maker(self.model, g_, t_ , self.single_qubit_mixer)

                        #loop through each starting state and find probability of outcomes
                        Q = np.zeros((2 ** self.n_spins, 2 ** self.n_spins))
                        energies = []
                        
                        #loop through starting states
                        for j, s in enumerate(self.S):
                            probs = CM.get_state_obtained_probs(s)
                            #allocate probs accordingly
                            Q[j, :] = probs

                        Q_ +=Q

                else:
                    #print("finally hereeeeee")
                    # just this value for now... it's a little rough...
                    
                    #n_ty_samples = abs(self.n_spins- self.max_qubits+1)*(max(self.time) - min(self.time))
                    #n_ty_samples = (2**n_spins-2**(max_qubits+1))*(max(time) - min(time))
                    #print(str(n_ty_samples)+" n_ty_samples!")
                    #model.model_summary()
                    #plt.show()

                    
                    #ie scaling:    #  linearly with Coarse graining       # linearly with time range         # linearly with gamma range             
                    n_ty_samples =int(abs(self.n_spins- self.max_qubits+1)*(np.max([1,(max(self.time) - min(self.time))])*np.max([1,(max(self.gamma*5) - min(self.gamma*5))]))*self.brute_sample_multiplier)
                    
                    times = np.random.randint(min(self.time), max(self.time),size= (n_ty_samples))
                    gammas = np.random.randint(int(min(self.gamma)*100),int(max(self.gamma)*100),size= (n_ty_samples))/100
                    
                    # print(times)
                    # print(gammas)
                    Q_ = np.zeros((2 ** self.n_spins, 2 ** self.n_spins))
                    #print(Q_)

                    for i in range(n_ty_samples):
                        t_ = times[i]
                        g_ = gammas[i]

                        Q__ = np.zeros((2 ** self.n_spins, 2 ** self.n_spins))
                        #loop through starting states
                        for k, s in enumerate(self.S):
                            S_, probs = self.sample_transitions_CG_probs_row(s, self.max_qubits, g_, t_)
                            S_ = [int(''.join(map(str, row)), 2) for row in S_]

                            #allocate probs accordingly
                            for j in range(len(S_)):
                                Q__[k, S_[j]] = probs[j]
                        Q_ += Q__
                    Q = Q_

                row_sums = Q_.sum(axis=1)
                row_sums[row_sums == 0] = 0.0000000001
                Q = Q_ / row_sums[:, np.newaxis]


        elif type(self.time) is tuple or type(self.gamma) is tuple:
            print("I havent coded time or gamma being varied, only both")
            print("If you get this message you can put your time or gamma in a tuple with min  = max, and it should work but I havent yet checked")
        else:
            print("time or gamma is of the wrong format/a format I haven't coded yet")

        return Q

    def find_acceptance_matrix(self)-> np.ndarray:
        if self.S is None:
            self.S = [''.join(i) for i in itertools.product('01', repeat=self.n_spins)]

        mu = []
        for i in range(2**self.n_spins):
            energy_s = self.model.get_energy(self.S[i])  #
            mu.append(energy_s)  # E(s')-E(s)
            #exp_factor = np.exp(-delta_energy / temperature)
        mu = np.array(mu)
        E = self.vectorized_find_E(mu)
        np.fill_diagonal(E,0)
        return E


    """
    # the following works, but has been upgraded elsewhere under the same name to allow for multiple CG samples to be taken

    def sample_transitions_CG_probs_row(self,s, n, gamma, time,choices = None):
        # find what state s' you obtain given s
        #is a list of all s and all s' (given the spins that aren't specified)
        current_state = s
        n_change_spins = n


        # decide which indexes to use
        full_index = np.arange(0, self.model.num_spins)

        #current bitstring as an array
        BIT_STRING_ARR = np.fromstring(current_state, 'u1') - ord('0')

        #The choice of spins that will be changed (index
        if choices is None:
            #random if not given
            choices = np.sort(self.find_subset(self.n_spins, n_change_spins))
        
        
        

        #current state of bitdrings to e changed
        change_bitstring = BIT_STRING_ARR[choices]



        mask = np.isin(full_index, choices, invert=True)

        #spins that wont be changes
        non_choices = full_index[mask]



        #defines the partial model post course grain
        partial_J = np.delete(self.model.J, non_choices, axis=0)
        partial_J = np.delete(partial_J, non_choices, axis=1)
        partial_h = np.delete(self.model.h, non_choices, axis=0)

        # define new model post course grain
        partial_model = IsingEnergyFunction(partial_J, partial_h, name="partial model")
        #partial_model.model_summary()
        #plt.show()
        #print(change_bitstring)
        c_btstring = ''.join(map(str, change_bitstring))


        CM = Circuit_Maker(partial_model,gamma,time, self.single_qubit_mixer)
        probs = CM.get_state_obtained_probs(c_btstring)


        S_ = [''.join(i) for i in itertools.product('01', repeat=partial_model.num_spins)]
        #print("S_")
        S_ = np.array([[int(bit) for bit in bitstring] for bitstring in S_])


        #put in all the original states for unchanged spins
        S_final = np.zeros((len(probs),self.n_spins), dtype = int)
        for count, nc in enumerate(non_choices):
            S_final[:,nc] = BIT_STRING_ARR[nc]

        #put in the possible states for the changing spins in the correct order
        for count_1, s_ in enumerate(S_):
            for count_2, c in enumerate(choices):
                S_final[count_1,c] = s_[count_2]

        return S_final, probs"""
        
    
    def sample_transitions_CG_probs_row(self ,s, n, gamma, time, choices = None):
        #now takes multiple CG samples
        # find what state s' you obtain given s
        #is a list of all s and all s' (given the spins that aren't specified)
        current_state = s
        n_change_spins = n


        # decide which indexes to use
        full_index = np.arange(0, self.model.num_spins)

        #current bitstring as an array
        BIT_STRING_ARR = np.fromstring(current_state, 'u1') - ord('0')

        #The choice of spins that will be changed (index
        if choices is None:
            #random if not given
            choices = np.sort(self.find_subset(self.n_spins, n_change_spins))
        
        all_choices = []
        #DO CG_sample_number number of quantum evaluations of max_qubits number of spins
        if self.CG_sample_number > 1:
            print("Optimised partial model doesn't work for chunking")
            print("doing non optimised (naive)")
        for i in range(0,self.CG_sample_number):
            
            #print(i)
            #find the next selection of choices to evaluate
            choices_i = []
            for i_c in choices:
                if i == 0:
                    nxt = i_c
                else:
                    nxt = i_c + self.max_qubits
                if nxt >= self.n_spins:
                    nxt = nxt-self.n_spins
                choices_i.append(nxt)
                
            choices = np.sort(choices_i)
            
            change_bitstring = BIT_STRING_ARR[choices]
            
            if self.CG_sample_number > 1 or self.naive == True:
                partial_model = self.define_partial_model(choices, full_index)
            else:
                partial_model = self.define_accurate_partial_model(choices, full_index, current_state)

            
            
            c_btstring = ''.join(map(str, change_bitstring))
            
            #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            #partial_E = partial_model.get_energy(c_btstring)
            #print("Partial_E: "+str(partial_E))
            

            CM = Circuit_Maker(partial_model,gamma,time, self.single_qubit_mixer)
            probs = CM.get_state_obtained_probs(c_btstring)


            S_ = [''.join(i) for i in itertools.product('01', repeat=partial_model.num_spins)]
            #print("S_")
            S_ = np.array([[int(bit) for bit in bitstring] for bitstring in S_])

        
            # Do the same with probs array
            if i == 0:
                
                probs_final = probs
                S_final = np.zeros((len(probs),self.n_spins), dtype = int)
                #put in the possible states for the changing spins in the correct order
                for count_1, s_ in enumerate(S_):
                    for count_2, c in enumerate(choices):
                        S_final[count_1,c] = s_[count_2]
            else:

                S_final = np.tile(S_final,(2**len(choices),1))
                probs_final = np.tile(probs_final,(2**len(choices)))
                for count_1, s_ in enumerate(S_):
                    S_final[(count_1*2**self.CG_sample_number):(count_1*2**self.CG_sample_number+2**self.CG_sample_number),choices] = s_
                    probs_final[(count_1*2**self.CG_sample_number):(count_1*2**self.CG_sample_number+2**self.CG_sample_number)] *= probs[count_1]
            all_choices.append(choices)
                
                
                

        #put in all the original states for unchanged spins
        possible_choices = np.arange(0,self.n_spins)
        global_non_choices = possible_choices[np.invert(np.isin(possible_choices,all_choices))]

        for count, nc in enumerate(global_non_choices):
            S_final[:,nc] = BIT_STRING_ARR[nc]
        
        return S_final, probs_final
    
    
    def define_partial_model(self,choices,full_index):
        #current state of bitstrings to be changed
        

        mask = np.isin(full_index, choices, invert=True)

        #spins that wont be changes
        non_choices = full_index[mask]

        #defines the partial model post course grain
        partial_J = np.delete(self.model.J, non_choices, axis=0)
        partial_J = np.delete(partial_J, non_choices, axis=1)
        partial_h = np.delete(self.model.h, non_choices, axis=0)

        # define new model post course grain
        partial_model = IsingEnergyFunction(partial_J, partial_h, name="partial model")
        

        return partial_model
    
    def define_accurate_partial_model(self,choices,full_index,current_state):
        #current state of bitstrings to be changed
        
        
        #mask = np.isin(full_index, choices, invert=True)
        mask = np.delete(np.copy(full_index),choices)
        #spins that wont be changed
        non_choices = full_index[mask]


        #defines the partial model post course grain
        partial_J = np.delete(self.model.J, non_choices, axis=0)
        partial_J = np.delete(partial_J, non_choices, axis=1)
        partial_h = np.delete(self.model.h, non_choices, axis=0)

        #print("current_state: "+str(current_state))
        #print("choices: "+str(choices))

        #print("partial_h: "+str(partial_h))

        #cnt is just to make undexing of choices easier
        cnt = 0

        #This is incredibly inefficient
        """
        for n in range(self.n_spins):
            if n in choices:
                for m in range(self.n_spins):
                    if m not in choices:
                    #Both J and h components of the removed spins contribute to the field
                        if int(current_state[m]) == 0:
                            partial_h[cnt] += (-1)*(self.model.J[n,m])
                        elif int(current_state[m]) == 1:
                            partial_h[cnt] += (1)*(self.model.J[n,m])#self.model.h[m]+
                        else:
                            print("something has gone wrong here")
                    # The partial_J is unchanged as the removes s is constant so the J sum in the hamiltonian (ie the matrix element (n,m)) 
                    # only has one variable, which is the value of the non-removed spin.
                    # This means it can be applied to the field, which is cheaper.
                cnt +=1
        """        
        for n in choices:
            for m in non_choices:
                #Both J and h components of the removed spins contribute to the field
                #print(current_state[m])
                #print(type(current_state[m]))
                if int(current_state[m]) == 0:
                    partial_h[cnt] += (-1)*(self.model.J[n,m])
                elif int(current_state[m]) == 1:
                    partial_h[cnt] += (1)*(self.model.J[n,m])#self.model.h[m]+
                else:
                    print("something has gone wrong here")
                # The partial_J is unchanged as the removes s is constant so the J sum in the hamiltonian (ie the matrix element (n,m)) 
                # only has one variable, which is the value of the non-removed spin.
                # This means it can be applied to the field, which is cheaper.
            cnt +=1

        # define new model post course grain
        partial_model = IsingEnergyFunction(partial_J, partial_h, name="partial model")


        return partial_model
    
    
    """
    
    # the following works, but has been upgraded elsewhere under the same name to allow for multiple CG samples on one lattice update
    def sample_transitions_CG_binary(self,s , n, gamma, time):
        # find what state s' you obtain given s
        #is a list of all s and all s' (given the spins that aren't specified)
        current_state = s
        n_change_spins = n

        
        # decide which indexes to use
        full_index = np.arange(0, self.n_spins)

        #current bitstring as an array
        BIT_STRING_ARR = np.fromstring(current_state, 'u1') - ord('0')

        #The choice of spins that will be changed (index
        choices = np.sort(self.find_subset(self.n_spins, n_change_spins))
        

        
        #current state of bitsrings to e changed
        change_bitstring = BIT_STRING_ARR[choices]



        mask = np.isin(full_index, choices, invert=True)

        #spins that wont be changes
        non_choices = full_index[mask]



        #defines the partial model post course grain
        partial_J = np.delete(self.model.J, non_choices, axis=0)
        partial_J = np.delete(partial_J, non_choices, axis=1)
        partial_h = np.delete(self.model.h, non_choices, axis=0)

        # define new model post course grain
        partial_model = IsingEnergyFunction(partial_J, partial_h, name="partial model")
        #partial_model.model_summary()
        #plt.show()
        #print(change_bitstring)
        c_btstring = ''.join(map(str, change_bitstring))
        CM = Circuit_Maker(partial_model,gamma,time, self.single_qubit_mixer)
        binary = CM.get_state_obtained_binary(c_btstring)
        

        #put in all the original states for unchanged spins       
        S_final = np.zeros(self.n_spins, dtype = int)
        for count, nc in enumerate(non_choices):
            S_final[nc] = BIT_STRING_ARR[nc]

        #put in the possible states for the changing spins in the correct order
        for count, c in enumerate(choices):
            S_final[c] = binary[count]
        return ''.join(map(str, S_final))"""


    



    def sample_transitions_CG_binary(self,s , n, gamma, time):
        # same as sample_transitions_CG_binary_ but can use multiple sections of the lattice
        # find what state s' you obtain given s
        # is a list of all s and all s' (given the spins that aren't specified)
        current_state = s
        n_change_spins = n

        
        # decide which indexes to use
        full_index = np.arange(0, self.n_spins)

        #current bitstring as an array
        BIT_STRING_ARR = np.fromstring(current_state, 'u1') - ord('0')

        #The choice of spins that will be changed (index
        orig_choices = self.find_subset(self.n_spins, n_change_spins)
        choices = np.sort(orig_choices)
        

        

        #DO CG_sample_number number of quantum evaluations of max_qubits number of spins
        for i in range(0,self.CG_sample_number):
            
            #find the next selection of choices to evaluate
            
            #if normal number of choices
            if self.sample_sizes[i] == self.max_qubits:
                
                choices_i = []
                for i_c in choices:
                    if i == 0:
                        nxt = i_c
                    else:
                        nxt = i_c + self.max_qubits
                    if nxt >= self.n_spins:
                        nxt = nxt-self.n_spins
                    choices_i.append(nxt)
                    
            #if not last
            # Unsure what this did.. have taken it out
            
            #elif i != self.CG_sample_number-1:
            #    choices_i = []
            #    print("hiii")
            #    for cnt, i_c in enumerate(choices):
            #         if i == 0:
            #             nxt = i_c
            #        else:
            #            nxt = i_c + self.max_qubits
            #        if nxt >= self.n_spins:
            #            nxt = nxt-self.n_spins
            #        choices_i.append(nxt)
            
            #if last iter (not normal numbe rof choices)
            elif i == self.CG_sample_number-1:
                choices_i = []
                for l in range(1,self.sample_sizes[-1]+1):
                    nxt = orig_choices[0]-l
                    if nxt < 0:
                        nxt = nxt+self.n_spins
                    choices_i.append(nxt)
            else:
                print("ahh this is why")
                
            
            choices = np.sort(choices_i)

            
            #current state of bitsrings to be changed
            change_bitstring = BIT_STRING_ARR[choices]
            #mask = np.isin(full_index, choices, invert=True)
            mask = np.delete(np.copy(full_index),choices) #much quicker than isin, and full index is only n long so space wise it shouldnt be an issue
            #spins that wont be changes
            non_choices = full_index[mask]

            
            partial_model = self.define_accurate_partial_model(choices, full_index, current_state)


            c_btstring = ''.join(map(str, change_bitstring))
            CM = Circuit_Maker(partial_model,gamma,time, self.single_qubit_mixer)
            binary = CM.get_state_obtained_binary(c_btstring)
            
            if i ==0:
                #put in all the original states for unchanged spins       
                S_final = np.zeros(self.n_spins, dtype = int)
                for count, nc in enumerate(non_choices):
                    S_final[nc] = BIT_STRING_ARR[nc]
                    
            else:
                #put in states from last iteration
                S_final_ = np.zeros(self.n_spins, dtype = int)
                for count, nc in enumerate(non_choices):
                    S_final_[nc] = S_final[nc]
                S_final = S_final_

            #put in the possible states for the changing spins in the correct order
            for count, c in enumerate(choices):
                S_final[c] = binary[count]
            current_state = S_final
            


        
        #print("ham dist")
        #print(BIT_STRING_ARR-S_final)
        #print(np.sum(abs(BIT_STRING_ARR-S_final)))
        return ''.join(map(str, S_final))



    def find_subset(self, arr_len, n, ind = None):
        # n is subset size
        # subset must all be neighbours (only works 2D atm)

        # index array
        ar = np.arange(0, arr_len)
        if ind == None:
            r = np.random.randint(0, arr_len + 1)
        else:
            r = ind
        ar = np.roll(ar, r, axis=0)

        # take the first n sections of the rearranged index array
        return ar[0:n]

    def integertobinary(self,L):        
        y = []
        for n in L:
            z = []
            while(n>0):
                a=n%2
                z.append(a)
                n=n//2
            z.reverse()
            y.append(z)
        return y

    def test_probs(self, 
        energy_s: float, energy_sprime: float) -> float:
        """
        Accepts the state "sprime" with probability A ( i.e. min(1,exp(-(E(s')-E(s))/ temp) )
        and s_init with probability 1-A.
        """
        delta_energy = energy_sprime - energy_s  # E(s')-E(s)
        exp_factor = np.exp(-delta_energy / self.temp)

        return exp_factor

    def vectorized_find_E(self,mu)-> float:
        # Assuming test_probs is a function that calculates exp_factor
        exp_factor = np.vectorize(self.test_probs)(mu[:, None], mu)
        E = np.minimum(1, exp_factor)
        return E

    def generate_combinations(self, total_bits, chosen_bits):
        def backtrack(start, path):
            if len(path) == chosen_bits:
                combinations.append(path)
                return
            for i in range(start, total_bits):
                backtrack(i + 1, path + [i])

        combinations = []
        backtrack(0, [])
        return combinations
    
def test_accept(
    energy_s: float, energy_sprime: float, temperature: float = 1.
) -> MCMCState:
    """
    Accepts the state "sprime" with probability A ( i.e. min(1,exp(-(E(s')-E(s))/ temp) )
    and s_init with probability 1-A.
    """
    delta_energy = energy_sprime - energy_s  # E(s')-E(s)
    exp_factor = np.exp(-delta_energy / temperature)
    acceptance = min(
        1, exp_factor
    )  # for both QC case as well as uniform random strategy, the transition matrix Pij is symmetric!

    return acceptance > np.random.rand()

def get_random_state(num_spins: int) -> str:
    """
    Returns s' , obtained via uniform transition rule!
    """
    num_elems = 2 ** (num_spins)
    #next_state = np.random.randint(
    #    0, num_elems
    #)  # since upper limit is exclusive and lower limit is inclusive
    next_state = random.randrange(0,num_elems,1)
    bin_next_state = f"{next_state:0{num_spins}b}"
    return bin_next_state
