##########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
import itertools
from typing import Optional
from tqdm import tqdm
from .basic_utils import MCMCChain, MCMCState
import scipy
import random







class MCMC_Classical:
    #class to set up the Markov Chain Monte Carlo,
    #also allows for spectral gap to be calculated
    
    def __init__(self, model, temp, Q= None, A= None, method = "uniform", cluster_size = None):


        self.Q = Q
        self.A =A
        self.model = model
        self.temp = temp
        self.beta = 1/self.temp
        self.n_spins = model.num_spins
        self.method = method
        self.S = None
        if self.method == "uniform":
            self.find_prob_matrix = self.find_prob_matrix_uniform
            self.update = self.update_uniform
        elif self.method == "local":
            self.find_prob_matrix = self.find_prob_matrix_local
            self.update = self.update_local
        elif self.method == "random_local":
            self.find_prob_matrix = self.find_prob_matrix_random_local
            self.update = self.update_random_local
            self.cluster_size = cluster_size
            if self.cluster_size == self.n_spins:
                print("this is essentially uniform as cluster size is same as number of spins")
                self.find_prob_matrix = self.find_prob_matrix_uniform
                self.update = self.update_uniform
        else:
            print("method name is incorrect. Choose from: 'uniform', 'local' or 'random_local'")
                
            
        

    def update_uniform(self,current_state):
        try:
            s_prime = get_random_state(len(current_state.bitstring))
        except:
             s_prime = get_random_state(len(current_state))
        return s_prime
        
    def update_local(self,current_state):
        # find what state s' you obtain given s
        
        #The choice of spins that will be changed (index
        choices = np.sort(self.find_subset(self.n_spins, 1))
        try:
            c_s = list(current_state.bitstring)
        except:
            c_s = list(current_state)
        #put in the possible states for the changing spins in the correct order
        for count, c in enumerate(choices):
            c_s[c] = str(int(c_s[c]) ^ 1)
 
        return ''.join(c_s)

        
    
    def update_random_local(self,current_state):
        # find what state s' you obtain given s
        
        #The choice of spins that will be changed (index
        choices = np.sort(self.find_subset(self.n_spins, self.cluster_size))
        binary = np.random.randint(0,2,size = len(choices))
        c_s = list(current_state.bitstring)
        #put in the possible states for the changing spins in the correct order
        for count, c in enumerate(choices):
            c_s[c] = str(binary[count])
        
        return ''.join(c_s)
    
    
    def find_prob_matrix_uniform(self):

        Q = np.ones((2**self.n_spins,2**self.n_spins))/(self.n_spins**2-1)
        row_sums = Q.sum(axis=1)
        Q = Q / row_sums[:, np.newaxis]
        return Q
    
    def find_prob_matrix_local(self):
        if self.S is None:
            self.S = [''.join(i) for i in itertools.product('01',repeat=self.n_spins)]

        Q = np.zeros((2 ** self.n_spins, 2 ** self.n_spins))


        #loop throguh and find the difference in bitstrings.
        #When the ith bitstring is different (by a valua of 1) from the jth bitstring add 1 to Q[i,j]
        for i in range(2 ** self.n_spins):
            for j in range(2 ** self.n_spins):
                sm = 0
                for l in range(self.n_spins):
                    sm += abs(int(self.S[i][l]) - int(self.S[j][l]))

                # ie if the number of different strings is the size of the cluster (= 1 for local)
                if sm == 1:
                    Q[i, j] = 1

        row_sums = Q.sum(axis=1)
        Q = Q / row_sums[:, np.newaxis]


        return Q


    def find_prob_matrix_random_local(self):
        
        if self.S is None:
            self.S = [''.join(i) for i in itertools.product('01', repeat=self.n_spins)]

        
        Q = np.zeros((2 ** self.n_spins, 2 ** self.n_spins))
        num_samples = ((2 ** self.n_spins) ** 2) * 10
        # loop throguh and find the difference in bitstrings.
        # When the ith bitstring is different from the jth bitstring ass 1 to [i,j]

        for i in range(2 ** self.n_spins):
            for n in range(num_samples // (2 ** self.n_spins)):
                # randomly pick cluster_size number of spins to randomly flip or not
                rand_indexes = np.random.choice(len(self.S[0]), size=self.cluster_size, replace=False)
                rand_flips = np.random.randint(0, 2, size=self.cluster_size)

                # pick out starting state
                s = [int(bi) for bi in self.S[i]]

                for m, r in enumerate(rand_indexes):
                    s[r] = s[r] ^ rand_flips[m]
                # print(s)
                # add one to Q[i, j] where j is the indexA corresponding to the bitstring we flipped to
                binary_s = ''.join(str(b) for b in s)
                Q[i, int(binary_s, 2)] += 1

        row_sums = Q.sum(axis=1)
        Q = Q / row_sums[:, np.newaxis]
        return Q




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





    def find_spec_gap_c(self):

        #check if Q is provided
        #useful for cycling through Temps
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
        verbose:bool = False,
        sample_frequency:int = 1):
        """
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
            initial_state = MCMCState(get_random_state(self.n_spins), accepted=True, pos = 0)
        
        else:
            initial_state = MCMCState(initial_state, accepted=True, pos = 0)
        
        current_state: MCMCState = initial_state
        energy_s = self.model.get_energy(current_state.bitstring)
        initial_state.energy = energy_s


        if verbose: print("starting with: ", current_state.bitstring, "with energy:", energy_s)



        

        mcmc_chain = MCMCChain([current_state], name= name)
        
        for i in tqdm(range(0, n_hops), desc='runnning classical MCMC steps . ..', disable= not verbose ):


            print(type(current_state))
            print(current_state.bitstring)            
            s_prime = self.update(current_state)
            
            energy_sprime = self.model.get_energy(s_prime)
            accepted = test_accept(
                energy_s, energy_sprime, temperature=self.temp
            )
            #mcmc_chain.add_state(MCMCState(s_prime, accepted, energy_sprime, pos = i))


            if accepted:
                current_state = MCMCState(s_prime, accepted, energy_s, pos = i)#mcmc_chain.current_state
                energy_s = energy_sprime#self.model.get_energy(current_state.bitstring)
                
                
            
            if i//sample_frequency == i/sample_frequency:
                mcmc_chain.add_state(MCMCState(current_state.bitstring, accepted, energy_s, pos = i))

        return mcmc_chain


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
    #    0, num_elems, dtype = int64)  # since upper limit is exclusive and lower limit is inclusive
    next_state = random.randrange(0,num_elems,1)
    #next_state = np.random.randint(0,num_elems,1, dtype = float)
    bin_next_state = f"{next_state:0{num_spins}b}"
    return bin_next_state
