##########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
import itertools
from typing import Optional
from tqdm import tqdm
from .basic_utils import MCMCChain, MCMCState
# from .prob_dist import *
from .energy_models import IsingEnergyFunction
from .classical_mcmc_routines import test_accept, get_random_state
import scipy
import random

#from .Circuit_Maker_qulacs import Hamming_Circuit_Maker as Circuit_Maker
from .Circuit_Maker_qulacs import Circuit_Maker




class MCMC_qulacs:
    """
    Class to set up the Quantum-enhanced Markov Chain Monte Carlo,
    allowing for Coarse Graining and spectral gap calculation.
    
    """
    
    def __init__(self, model, gamma, time, temp, max_qubits=None, CG_sample_number=1, Q=None, A=None, single_qubit_mixer=True, brute_sample=False, brute_sample_multiplier=1, naive=False, noise_model = None, noise_prob_one_qubit = 0, noise_prob_two_qubit = 0):
        """
        Initializes an instance of the CGQeMCMC class.

        Args:
            model (Model): The model to be simulated.
            gamma (float): The gamma parameter.
            time (float): The time parameter.
            temp (float): The temperature parameter.
            max_qubits (int, optional): The maximum number of qubits to use ie. the size of the group in paper. Defaults to None.
            CG_sample_number (int, optional): The number of CG samples to take ie. the number of groups to evaluate (see paper). Defaults to 1.
            Q (None, optional): Optional input of Q Matrix (if, for example it has already been computed elsewhere). This is useful if one wishes to cycle through different temperatures without repeatedly finding Q. Defaults to None.
            A (None, optional): Optional input of A Matrix (if, for example it has already been computed elsewhere). This is useful if one wishes to cycle through different temperatures without repeatedly finding A. Defaults to None.
            single_qubit_mixer (bool, optional): Flag indicating whether to use single-qubit mixer. Defaults to True.
            brute_sample (bool, optional): Flag indicating whether to brute sample the Q matrix. Defaults to False.
            brute_sample_multiplier (int, optional): The multiplier for brute sampling. Defaults to 1. num_samples = ((2**n_spins)**2)*brute_sample_multiplier
            naive (bool, optional): Flag indicating whether to use naive approach (see paper). Defaults to False.
        """
        self.Q = Q
        self.noise_prob_one_qubit = noise_prob_one_qubit
        self.noise_prob_two_qubit = noise_prob_two_qubit
        self.noise_model = noise_model
        self.brute_sample = brute_sample
        self.brute_sample_multiplier = brute_sample_multiplier
        self.A = A
        self.model = model
        self.gamma = gamma
        self.time = time
        self.temp = temp
        self.beta = 1 / self.temp
        self.n_spins = model.num_spins
        self.delta_time = 0.8
        self.max_qubits = max_qubits
        self.CG_sample_number = CG_sample_number
        self.naive = naive

        #sort out how many quantum computations to do,a dn what sizes to do
        avg_sample_size = self.n_spins / self.CG_sample_number
        if avg_sample_size == self.max_qubits:
            self.sample_sizes = np.ones(self.CG_sample_number) * self.max_qubits
        elif avg_sample_size <= self.max_qubits:
            self.sample_sizes = np.ones(self.CG_sample_number, dtype=int) * self.max_qubits
            self.sample_sizes[-1] = self.n_spins - np.sum(self.sample_sizes[:-1])
        else:
            self.sample_sizes = [self.max_qubits,]
            
            
            
        print("Quantum computations for this Ising model problem will be of qubit sizes: " +str(self.sample_sizes))

        self.single_qubit_mixer = single_qubit_mixer
        self.S = None

        if self.max_qubits == self.model.num_spins or self.max_qubits is None:
            self.course_graining = False
        else:
            self.course_graining = True

        

    def find_spec_gap_q(self):
        """
        Calculates the spectral gap of the probability matrix Q.

        Returns:
            float: The spectral gap of the probability matrix Q.
        """
        # Check if Q is provided
        # Useful for cycling through Temps
        self.model.get_all_energies()
        if self.Q is None:
            self.Q = self.find_prob_matrix()

        # Check if A is provided
        # Unsure when this will be useful
        if self.A is None:
            self.A = self.find_acceptance_matrix()

        P = np.multiply(self.Q, self.A)

        # Account for rejected swaps to add to s = s' matrix element
        for i in range(P.shape[0]):
            s = np.sum(P[i, :])
            P[i, i] = 1 - s

        # Normalize by row
        row_sums = P.sum(axis=1)
        P = P / row_sums[:, np.newaxis]

        # Find eigenvalues
        e_vals, e_vecs = scipy.linalg.eig(P)
        e_vals = np.flip(np.sort(abs(e_vals)))

        # Find spectral gap
        delta = e_vals[1]
        delta = 1 - delta

        self.delta = delta

        return delta


    def run(self,
        n_hops: int,
        initial_state: Optional[str] = None,
        name:str = "(CG)QeMCMC",
        verbose:bool = False,
        sample_frequency:int = 1):
        """
        Runs the quantum MCMC algorithm for a specified number of hops.

        Args:
            n_hops (int): The number of hops to perform in the MCMC algorithm.
            initial_state (Optional[str], optional): The initial state for the MCMC algorithm. If not provided, a random state will be generated. Defaults to None.
            name (str, optional): The name of the MCMC chain. Defaults to "quMCMC".
            verbose (bool, optional): Whether to print verbose output during the algorithm execution. Defaults to False.
            sample_frequency (int, optional): The frequency at which to sample states. Defaults to 1.

        Returns:
            MCMCChain: The MCMC chain containing the states collected during the algorithm execution.
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
            



            if accepted:
                current_state = MCMCState(s_prime, accepted, energy_sprime, pos = i) #mcmc_chain.current_state# s_prime#mcmc_chain.current_state#MCMCState(s_prime, accepted, energy_sprime, pos = i) #mcmc_chain.current_state
                energy_s = energy_sprime #self.model.get_energy(current_state.bitstring)

            
            if i//sample_frequency == i/sample_frequency and i != 0:
                mcmc_chain.add_state(MCMCState(current_state.bitstring, accepted, energy_s, pos = i))
        
        return mcmc_chain
    
    
    def run_many_hops_constant_s(self,
        n_hops: int,
        initial_state: np.ndarray,
        name:str = "(CG)QeMCMC",
        verbose:bool = False):
        """
        Runs the quantum MCMC algorithm for a specified number of hops.

        Args:
            n_hops (int): The number of hops to perform in the MCMC algorithm.
            initial_state (Optional[str], optional): The initial state for the MCMC algorithm. If not provided, a random state will be generated. Defaults to None.
            name (str, optional): The name of the MCMC chain. Defaults to "quMCMC".
            verbose (bool, optional): Whether to print verbose output during the algorithm execution. Defaults to False.
            sample_frequency (int, optional): The frequency at which to sample states. Defaults to 1.

        Returns:
            MCMCChain: The MCMC chain containing the states collected during the algorithm execution.
        """



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

        energies = []
        for i in tqdm(range(0, n_hops), desc='runnning quantum constant s steps ...', disable= not verbose ):
            s_prime = self.get_s_prime(current_state, gamma[i], time[i])
            energy_sprime = self.model.get_energy(s_prime)
            energies.append(energy_sprime)  
            
        return energies, energy_s
    
    def get_s_prime(self, current_state, g, t):
        """
        Returns the next state s_prime based on the current state, g, and t.

        Parameters:
        current_state (str): The current state.
        g (float or tuple): The value of g. If it's a tuple, a random value is generated between the minimum and maximum values of the tuple.
        t (int or tuple): The value of t. If it's a tuple, a random integer is generated between the minimum and maximum values of the tuple.

        Returns:
        str: The next state s_prime.
        """
        if g is None:
            if type(self.gamma) is tuple:
                g = np.round(np.random.uniform(low=min(self.gamma), high=max(self.gamma), size=1), decimals=6)
            else:
                g = self.gamma
        if t is None:
            if type(self.time) is tuple:
                t = np.random.randint(low=np.min(self.time), high=np.max(self.time), size=1)
            else:
                t = self.time

        if not self.course_graining:
            CM = Circuit_Maker(self.model, g, t, self.single_qubit_mixer, noise_model = self.noise_model, noise_prob_one_qubit = self.noise_prob_one_qubit, noise_prob_two_qubit = self.noise_prob_two_qubit)
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
        """
        Calculates the probability matrix Q based on the given parameters.

        Returns:
            Q (numpy.ndarray): The probability matrix.
        """
        if self.S is None:
            self.S = [''.join(i) for i in itertools.product('01', repeat=self.n_spins)]
        
        if type(self.time) is int and type(self.gamma) is float:
            # If time and gamma are single
            if self.brute_sample:
                # If brute sample
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
                        CM = Circuit_Maker(self.model, self.gamma, self.time, self.single_qubit_mixer, noise_model = self.noise_model, noise_prob_one_qubit = self.noise_prob_one_qubit, noise_prob_two_qubit = self.noise_prob_two_qubit)
                        probs = CM.get_state_obtained_probs(s)
                        Q[i, :] += probs
                    

                else:
                    #If coarse grained
                    
                    Q = np.zeros((2 ** self.n_spins, 2 ** self.n_spins))
                    all_combs = self.generate_combinations(self.n_spins, self.max_qubits)
                    for k, s in enumerate(self.S):
                        for comb in all_combs:
                            print("sample_transitions_CG_probs_row doesn't work with input choices.")
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

                n_ty_samples =int(2**self.n_spins*self.brute_sample_multiplier)
                times = np.random.randint(min(self.time), max(self.time),size= (n_ty_samples))
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
                            print("Weird error happened again!")
                row_sums = Q.sum(axis=1)
                row_sums[row_sums == 0] = 0.0000000001
                Q = Q / row_sums[:, np.newaxis]
                    
                    
            else:
                if self.max_qubits == None or self.max_qubits == self.n_spins:

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
                        CM = Circuit_Maker(self.model, g_, t_ , self.single_qubit_mixer, noise_model = self.noise_model, noise_prob_one_qubit = self.noise_prob_one_qubit, noise_prob_two_qubit = self.noise_prob_two_qubit)

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

                    #ie scaling:    #  linearly with Coarse graining       # linearly with time range         # linearly with gamma range             
                    n_ty_samples =int(abs(self.n_spins- self.max_qubits+1)*(np.max([1,(max(self.time) - min(self.time))])*np.max([1,(max(self.gamma*5) - min(self.gamma*5))]))*self.brute_sample_multiplier)
                    
                    times = np.random.randint(min(self.time), max(self.time),size= (n_ty_samples))
                    gammas = np.random.randint(int(min(self.gamma)*100),int(max(self.gamma)*100),size= (n_ty_samples))/100

                    Q_ = np.zeros((2 ** self.n_spins, 2 ** self.n_spins))

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
            print("If you get this message you can put your time or gamma in a tuple with min  =  max, and it should work but I haven't yet checked")
        else:
            print("time or gamma is of the wrong format/a format I haven't coded yet")

        return Q

    def find_acceptance_matrix(self) -> np.ndarray:
        """
        Calculates and returns the acceptance matrix for the CGQeMCMC algorithm.

        Returns:
            np.ndarray: The acceptance matrix.
        """
        if self.S is None:
            self.S = [''.join(i) for i in itertools.product('01', repeat=self.n_spins)]

        mu = []
        for i in range(2**self.n_spins):
            energy_s = self.model.get_energy(self.S[i])  #
            mu.append(energy_s)  # E(s')-E(s)
        mu = np.array(mu)
        E = self.vectorized_find_E(mu)
        np.fill_diagonal(E, 0)
        return E



        
    
    def sample_transitions_CG_probs_row(self, s, n, gamma, time, choices=None):
        """
        Sample transitions and calculate probabilities for a given row of the CGQeMCMC matrix.

        Args:
            s (str): The current state.
            n (int): The number of spins to change.
            gamma (float): The gamma parameter.
            time (float): The time parameter.
            choices (ndarray, optional): The indices of spins to change. If not provided, random spins will be chosen.

        Returns:
            tuple: A tuple containing the final states and their corresponding probabilities.

        Raises:
            None

        """

        current_state = s
        n_change_spins = n

        # decide which indexes to use
        full_index = np.arange(0, self.model.num_spins)

        # current bitstring as an array
        BIT_STRING_ARR = np.fromstring(current_state, 'u1') - ord('0')

        # The choice of spins that will be changed (index
        if choices is None:
            # random if not given
            choices = np.sort(self.find_subset(self.n_spins, n_change_spins))

        all_choices = []

        # DO CG_sample_number number of quantum evaluations of max_qubits number of spins
        if self.CG_sample_number > 1:
            print("Optimised partial model doesn't work for chunking")
            print("doing non optimised (naive)")

        for i in range(0, self.CG_sample_number):
            choices_i = []
            for i_c in choices:
                if i == 0:
                    nxt = i_c
                else:
                    nxt = i_c + self.max_qubits
                if nxt >= self.n_spins:
                    nxt = nxt - self.n_spins
                choices_i.append(nxt)

            choices = np.sort(choices_i)

            change_bitstring = BIT_STRING_ARR[choices]

            if self.naive == True:
                partial_model = self.define_partial_model(choices, full_index)
            else:
                partial_model = self.define_accurate_partial_model(choices, full_index, current_state)

            c_btstring = ''.join(map(str, change_bitstring))

            CM = Circuit_Maker(partial_model, gamma, time, self.single_qubit_mixer, noise_model = self.noise_model, noise_prob_one_qubit = self.noise_prob_one_qubit, noise_prob_two_qubit = self.noise_prob_two_qubit)
            probs = CM.get_state_obtained_probs(c_btstring)

            S_ = [''.join(i) for i in itertools.product('01', repeat=partial_model.num_spins)]
            S_ = np.array([[int(bit) for bit in bitstring] for bitstring in S_])

            if i == 0:
                probs_final = probs
                S_final = np.zeros((len(probs), self.n_spins), dtype=int)
                for count_1, s_ in enumerate(S_):
                    for count_2, c in enumerate(choices):
                        S_final[count_1, c] = s_[count_2]
            else:
                S_final = np.tile(S_final, (2 ** len(choices), 1))
                probs_final = np.tile(probs_final, (2 ** len(choices)))
                for count_1, s_ in enumerate(S_):
                    S_final[(count_1 * 2 ** self.CG_sample_number):(count_1 * 2 ** self.CG_sample_number + 2 ** self.CG_sample_number), choices] = s_
                    probs_final[(count_1 * 2 ** self.CG_sample_number):(count_1 * 2 ** self.CG_sample_number + 2 ** self.CG_sample_number)] *= probs[count_1]
            all_choices.append(choices)

        possible_choices = np.arange(0, self.n_spins)
        global_non_choices = possible_choices[np.invert(np.isin(possible_choices, all_choices))]

        for count, nc in enumerate(global_non_choices):
            S_final[:, nc] = BIT_STRING_ARR[nc]

        return S_final, probs_final
    
    
    def define_partial_model(self, choices, full_index):
        """
        Defines a partial model based on the given choices and full index.
        This is the naive approach to defining a partial model as explained in the paper.
        
        Parameters:
        choices (list): A list of indices representing the spins to be changed.
        full_index (list): A list of all indices representing the spins in the full model.

        Returns:
        partial_model (IsingEnergyFunction): A partial model post coarse-graining.

        """
        # current state of bitstrings to be changed
        mask = np.isin(full_index, choices, invert=True)

        # spins that won't be changed
        non_choices = full_index[mask]

        # defines the partial model post coarse-graining
        partial_J = np.delete(self.model.J, non_choices, axis=0)
        partial_J = np.delete(partial_J, non_choices, axis=1)
        partial_h = np.delete(self.model.h, non_choices, axis=0)

        # define new model post coarse-graining
        partial_model = IsingEnergyFunction(partial_J, partial_h, name="partial model")

        return partial_model
    
    def define_accurate_partial_model(self, choices, full_index, current_state):
        """
        Defines a partial model based on the given choices, full index, and current state.
        This is the improved approach as defined in the paper.
        Args:
            choices (list): List of indices representing the spins to be changed.
            full_index (ndarray): Array of all spin indices.
            current_state (str): Current state of bitstrings.

        Returns:
            partial_model (IsingEnergyFunction): Partial model post coarse-graining.

        Raises:
            None

        """

        # current state of bitstrings to be changed

        # mask = np.isin(full_index, choices, invert=True)
        mask = np.delete(np.copy(full_index), choices)
        # spins that won't be changed
        non_choices = full_index[mask]

        # defines the partial model post coarse-graining
        partial_J = np.delete(self.model.J, non_choices, axis=0)
        partial_J = np.delete(partial_J, non_choices, axis=1)
        partial_h = np.delete(self.model.h, non_choices, axis=0)

        # cnt is just to make indexing of choices easier
        cnt = 0

        for n in choices:
            for m in non_choices:
                # Both J and h components of the removed spins contribute to the field

                if int(current_state[m]) == 0:
                    partial_h[cnt] += (-1) * (self.model.J[n, m])
                elif int(current_state[m]) == 1:
                    partial_h[cnt] += (1) * (self.model.J[n, m])  # self.model.h[m]+
                else:
                    print("something has gone wrong here")
                # The partial_J is unchanged as the removed spin is constant so the J sum in the Hamiltonian (i.e., the matrix element (n,m))
                # only has one variable, which is the value of the non-removed spin.
                # This means it can be applied to the field, which is cheaper.
            cnt += 1

        # define new model post coarse-graining
        partial_model = IsingEnergyFunction(partial_J, partial_h, name="partial model",no_inits = True)

        return partial_model
    
    

    def sample_transitions_CG_binary(self, s, n, gamma, time):
        """
        Perform binary transitions sampling using the CGQeMCMC algorithm. This is used in "brute force" sampling of Q or in actual CGQeMCMC.

        Args:
            s (str): The initial state of the system.
            n (int): The number of spins to be changed in each transition.
            gamma (float): The parameter for the CGQeMCMC algorithm.
            time (float): The time parameter for the CGQeMCMC algorithm.

        Returns:
            str: The final state of the system after performing the transitions.

        Raises:
            None

        """
        # same as sample_transitions_CG_binary_ but can use multiple sections of the lattice
        # find what state s' you obtain given s
        # is a list of all s and all s' (given the spins that aren't specified)
        current_state = s
        n_change_spins = n

        # decide which indexes to use
        full_index = np.arange(0, self.n_spins)

        # current bitstring as an array
        BIT_STRING_ARR = np.fromstring(current_state, 'u1') - ord('0')

        # The choice of spins that will be changed (index
        orig_choices = self.find_subset(self.n_spins, n_change_spins)
        choices = np.sort(orig_choices)

        # DO CG_sample_number number of quantum evaluations of max_qubits number of spins
        for i in range(0, self.CG_sample_number):

            # find the next selection of choices to evaluate

            # if normal number of choices
            if self.sample_sizes[i] == self.max_qubits:

                choices_i = []
                for i_c in choices:
                    if i == 0:
                        nxt = i_c
                    else:
                        nxt = i_c + self.max_qubits
                    if nxt >= self.n_spins:
                        nxt = nxt - self.n_spins
                    choices_i.append(nxt)

            elif i == self.CG_sample_number - 1:
                choices_i = []
                for l in range(1, self.sample_sizes[-1] + 1):
                    nxt = orig_choices[0] - l
                    if nxt < 0:
                        nxt = nxt + self.n_spins
                    choices_i.append(nxt)
            else:
                print("a weird error has occured in sample_transitions_CG_binary")

            choices = np.sort(choices_i)

            # current state of bitsrings to be changed
            change_bitstring = BIT_STRING_ARR[choices]
            # mask = np.isin(full_index, choices, invert=True)
            mask = np.delete(np.copy(full_index), choices)  # much quicker than isin, and full index is only n long so space wise it shouldnt be an issue
            # spins that wont be changes
            non_choices = full_index[mask]

            partial_model = self.define_accurate_partial_model(choices, full_index, current_state)

            c_btstring = ''.join(map(str, change_bitstring))

            CM = Circuit_Maker(partial_model, gamma, time, self.single_qubit_mixer, noise_model = self.noise_model, noise_prob_one_qubit = self.noise_prob_one_qubit, noise_prob_two_qubit = self.noise_prob_two_qubit)
            binary = CM.get_state_obtained_binary(c_btstring)
            if i == 0:
                # put in all the original states for unchanged spins
                S_final = np.zeros(self.n_spins, dtype=int)
                for count, nc in enumerate(non_choices):
                    S_final[nc] = BIT_STRING_ARR[nc]

            else:
                # put in states from last iteration
                S_final_ = np.zeros(self.n_spins, dtype=int)
                for count, nc in enumerate(non_choices):
                    S_final_[nc] = S_final[nc]
                S_final = S_final_

            # put in the possible states for the changing spins in the correct order
            for count, c in enumerate(choices):
                S_final[c] = binary[count]
            current_state = S_final

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
