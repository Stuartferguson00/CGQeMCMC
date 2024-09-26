##########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
from typing import Optional
from tqdm import tqdm
from .basic_utils import MCMCChain, MCMCState
from .energy_models import IsingEnergyFunction
from .classical_mcmc_routines import get_random_state, test_accept


#from .Circuit_Maker_qulacs import Hamming_Circuit_Maker as Circuit_Maker
from .Circuit_Maker_qulacs import Circuit_Maker




class MCMC_qulacs:
    """
    Class to set up the Quantum-enhanced Markov Chain Monte Carlo allowing for Coarse Graining.
    
    """
    
    def __init__(self, model, gamma, time, temp, max_qubits=None, CG_sample_number=1, naive=False, delta_time = 0.8):
        #havent done type hinting yet
        """
        Initializes an instance of the CGQeMCMC class.

        Args:
            model (Model): The model to be simulated.
            gamma (float): The gamma parameter.
            time (float): The time parameter.
            temp (float): The temperature parameter.
            max_qubits (int, optional): The maximum number of qubits to use ie. the size of the group in paper. Defaults to None.
            CG_sample_number (int, optional): The number of CG samples to take ie. the number of groups to evaluate (see paper). Defaults to 1.
            naive (bool, optional): Flag indicating whether to use naive approach (see paper). Defaults to False.
        """
        
        self.model = model
        self.n_spins = model.num_spins
        
        self.gamma = gamma
        self.time = time
        self.delta_time = delta_time
        
        self.temp = temp
        self.beta = 1 / self.temp
        
        self.max_qubits = max_qubits
        self.CG_sample_number = CG_sample_number
        self.naive = naive

        # Sort out how many quantum computations to do and what sizes to do
        # Future versions should probably just take as input the sample_sizes list
        avg_sample_size = self.n_spins / self.CG_sample_number
        if avg_sample_size == self.max_qubits:
            self.sample_sizes = np.ones(self.CG_sample_number) * self.max_qubits
        elif avg_sample_size <= self.max_qubits:
            self.sample_sizes = np.ones(self.CG_sample_number, dtype=int) * self.max_qubits
            self.sample_sizes[-1] = self.n_spins - np.sum(self.sample_sizes[:-1])
        else:
            self.sample_sizes = [self.max_qubits,]

        print("Quantum computations for this Ising model problem will be of qubit sizes: " +str(self.sample_sizes))


        if self.max_qubits == self.model.num_spins or self.max_qubits is None:
            self.course_graining = False
        else:
            self.course_graining = True


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


        # Either get a random state or use initial state given
        if initial_state is None:
            initial_state = MCMCState(get_random_state(self.n_spins), accepted=True, pos = 0)
        else:
            initial_state = MCMCState(initial_state, accepted=True, pos = 0)
        
        #set initial state
        current_state: MCMCState = initial_state
        energy_s = self.model.get_energy(current_state.bitstring)
        initial_state.energy = energy_s


        if verbose: print("starting with: ", current_state.bitstring, "with energy:", energy_s)



        
        # Define chain
        mcmc_chain = MCMCChain([current_state], name= name)

        # Do MCMC
        for i in tqdm(range(0, n_hops), desc='running (CG)QeMCMC', disable= not verbose ):
            
            # Propose a new state
            s_prime = self.get_s_prime(current_state.bitstring)

            #Find energy of the new state
            energy_sprime = self.model.get_energy(s_prime)
            # Decide whether to accept the new state
            accepted = test_accept(energy_s, energy_sprime, temperature=self.temp)
            


            # If accepted, update current_state
            if accepted:
                current_state = MCMCState(s_prime, accepted, energy_sprime, pos = i)
                energy_s = energy_sprime

            # if time to sample, add state to chain
            if i//sample_frequency == i/sample_frequency and i != 0:
                mcmc_chain.add_state(MCMCState(current_state.bitstring, accepted, energy_s, pos = i))
            
            
        return mcmc_chain
    
    def get_s_prime(self, current_state):
        """
        Returns the next state s_prime based on the current state, g, and t.

        Args:
        current_state (str): The current state.
        g (float or tuple): The value of g. If it's a tuple, a random value is generated between the minimum and maximum values of the tuple.
        t (int or tuple): The value of t. If it's a tuple, a random integer is generated between the minimum and maximum values of the tuple.

        Returns:
        str: The next state s_prime.
        """
        # Should probably type check time and gamma in init not here 
        if type(self.gamma) is float or type(self.gamma) is int:
            g = self.gamma
        elif type(self.gamma) is tuple:
            g = np.round(np.random.uniform(low= min(self.gamma), high = max(self.gamma),size = 1), decimals=6)
        else:
            raise TypeError("gamma must be either a float or a tuple")
            
        if type(self.time) is int:
            t = self.time
        elif type(self.time) is tuple:
            t = np.random.randint(low= np.min(self.time), high = np.max(self.time),size = 1)
        else:
            raise TypeError("time must be either an int or a tuple")
        
        
        
        # Get s_prime
        if not self.course_graining:
            CM = Circuit_Maker(self.model, g, t)
            s_prime = CM.get_state_obtained_binary(current_state)
        else:
            s_prime = self.sample_transitions_CG_binary(current_state, self.max_qubits, g, t)

        return s_prime


    



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
        partial_model = IsingEnergyFunction(partial_J, partial_h, name="partial model", no_initial_states = True)

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
        partial_model = IsingEnergyFunction(partial_J, partial_h, name="partial model",no_initial_states = True)

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

            CM = Circuit_Maker(partial_model, gamma, time)
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


