###########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
import matplotlib.pyplot as plt
from .prob_dist import  value_sorted_dict, DiscreteProbabilityDistribution
from .basic_utils import plot_bargraph_desc_order
from typing import Union
from tqdm import tqdm
import itertools
import typing
###########################################################################################
## ENERGY MODEL ##
###########################################################################################

class IsingEnergyFunction:
    
    """
    A class to build the Ising Energy Function from self.
    Heavily modified from https://github.com/pafloxy/quMCMC to add functionality.
    Attributes:
    -----------
    negative_energy : bool
        Indicates if the energy should be negative.
    J : np.array
        Weight-matrix of the interactions between the spins.
    h : np.array
        Local field to the spins.
    S : list or None
        List of all possible states.
    all_energies : np.array or None
        Array of energies for all possible states.
    lowest_energy : float or None
        The lowest energy found.
    num_spins : int
        Number of spins in the system.
    alpha : float
        Scaling factor for the energy.
    name : str
        Name of the Ising model.
    initial_state : list
        List of initial states for simulations.
    Methods:
    --------
    __init__(J: np.array, h: np.array, name: str = None, negative_energy: bool = True, no_inits = False) -> None
        Initializes the IsingEnergyFunction with given parameters.
    get_energy(state: str) -> float
        Returns the energy of a given state.
    calc_an_energy(state: str) -> float
        Calculates the energy of a given state.
    get_all_energies() -> None
        Calculates and stores the energies of all possible states.
    get_lowest_energies(num_states: int) -> tuple[np.array, np.array]
        Returns the lowest energies and their degeneracies.
    find_lowest_values(arr: np.array, num_values: int = 5) -> tuple[np.array, np.array]
        Finds the lowest values in an array and their counts.
    get_lowest_energy() -> float
        Returns the lowest energy found.
    get_boltzmann_factor(state: Union[str, np.array], beta: float = 1.0) -> float
        Returns the un-normalized Boltzmann probability of a given state.
    get_boltzmann_factor_from_energy(E: float, beta: float = 1.0) -> float
        Returns the un-normalized Boltzmann probability for a given energy.

    
    """    

    def __init__(self, J: np.array, h: np.array, name:str = None, cost_function_signs:list = [-1,-1], no_initial_states = False) -> None:
        
        """
        Initialize the Ising model.
        Parameters:
            J (np.array): Weight-matrix of the interactions between the spins.
            h (np.array): Local field to the spins.
            name (str, optional): Name of the Ising model. Defaults to None.
            cost_function_signs (list, optional): List of two elements, the first element is the sign of the interaction term and the second element is the sign of the field term. Allows for the cost function to be flipped with respect to the standard Ising model. Defaults to [-1, -1].
            no_initial_states (bool, optional): If True, no initial states are stored for the model. Defaults to False.

        """

        # self.cost_function_signs allows for cost function to be flipped wrt to the standard Ising model
        self.cost_function_signs = cost_function_signs
        self.J = J
        self.h = h
        self.name = name 
        self.S = None
        self.lowest_energy  = None
        self.num_spins = len(h)
        self.alpha = np.sqrt(self.num_spins) / np.sqrt( sum([J[i][j]**2 for i in range(self.num_spins) for j in range(i)]) + sum([h[j]**2 for j in range(self.num_spins)])  )

        
        
        #100 optional optional states to use for fair starting state between different simulations
        if no_initial_states:
            self.initial_state = []
        else:
            self.initial_state = []
            for i in range(100): 
                self.initial_state.append(''.join(str(i) for i in np.random.randint(0, 2, self.num_spins, dtype = int)))





    def get_energy(self, state: str) -> float:
        """ 
        Returns the energy of a given state

        Args:
        state (str) : Configuration of spins for which the energy is required to be calculated.

        """
        if not isinstance(state,str):
            raise TypeError(f"State must be a string, but got {type(state)}")
        

        energy = self.calc_an_energy(state)
        
        return energy
    
            
    def calc_an_energy(self,state:str) -> float:
        """
        Calculate the energy of a given state.
        
        This function computes the energy of a given state based on the Ising model.
        The state is expected to be a string of "0"s and "1"s, which are converted to
        -1 and 1 respectively for the calculation.
        
        Args:
        state (str): A string representing the state, where each character is either "0" or "1".
        
        Returns
        float: The calculated energy of the given state.
        
        Raises:
        TypeError
            If the input state is not a string.
        """

        if not isinstance(state, str):
            raise TypeError(f"State must be a string, but got {type(state)}")
        
        
        state = np.array([-1 if elem == "0" else 1 for elem in state])
        
        #THIS ONLY WORKS IF THE INPUT IS NOT UPPER DIAGONAL.
        # self.cost_function_signs allows for cost function to be flipped wrt to the standard Ising model
        try:
            energy = self.cost_function_signs[0]* 0.5 * np.dot(state.transpose(), self.J.dot(state)) + self.cost_function_signs[1]* np.dot(self.h.transpose(), state)
        except Exception as e:
            print(f"Error calculating energy for state {state}: {e}")
            print("This error is generally caused when qulacs outputs a bitstring of 1 followed by n 0's for the state for some reason")
            energy = 10000
        


        return energy
    
    
    
    
    
    
    
    
    
    
    
    def get_all_energies(self) -> np.ndarray :
        """
        Calculate the energies for all possible spin states.
        This method generates all possible spin states for the system, calculates the energy for each state,
        and returns an array of these energies.
        Returns:
            np.ndarray: An array containing the energies of all possible spin states.
        """
        self.S = [''.join(i) for i in itertools.product('01', repeat=self.num_spins)]
        all_energies = np.zeros(len(self.S))
        for state in self.S:
            all_energies[int(state,2)] = self.calc_an_energy(state)
        return all_energies
            
            
    def get_lowest_energies(self,num_states:int) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the lowest energy states and their degeneracies.
        This method computes all possible energies and then finds the specified number
        of lowest energy states along with their degeneracies. Note that this method 
        is intended for small instances due to its brute-force nature, which is extremely 
        memory intensive and slow.
        Args:
            num_states (int): The number of lowest energy states to retrieve.
        Returns:
            Two numpy arrays:
                - The first array contains the lowest energy values.
                - The second array contains the degeneracies of the corresponding energy values.
        """
        #only to be used for small instances, it is just brute force so extremely memory intensive and slow
        all_energies = self.get_all_energies()

        #very slow (sorts whole array)  
        self.lowest_energies, self.lowest_energy_degeneracy = self.find_lowest_values(all_energies, num_values=num_states)

        return self.lowest_energies, self.lowest_energy_degeneracy
    
    def find_lowest_values(self, arr: np.ndarray, num_values: int = 5):
        """
        Find the lowest unique values in an array and their degeneracies.

        Args:
            arr (np.ndarray): The input array from which to find the lowest values.
            num_values (int, optional): The number of lowest unique values to find. Defaults to 5.

        Returns:
        tuple: A tuple containing two numpy arrays:
            - lowest_values (np.ndarray): The lowest unique values in the array.
            - degeneracy (np.ndarray): The counts of each of the lowest unique values.
        """
        # Count the occurrences of each value
        unique_values, counts = np.unique(arr, return_counts=True)
        # Sort the unique values and counts by value
        sorted_indices = np.argsort(unique_values)
        unique_values_sorted = unique_values[sorted_indices]
        counts_sorted = counts[sorted_indices]
        # Find the first num_values
        lowest_values = unique_values_sorted[:num_values]
        degeneracy = counts_sorted[:num_values]
        return lowest_values, degeneracy
    
    def get_lowest_energy(self):
        """
        Calculate and return the lowest energy from all possible energies.
        This method uses a brute force approach to find the lowest energy, 
        making it extremely memory intensive and slow. It is recommended 
        to use this method only for small instances.
        Returns:
            float: The lowest energy value.
        Notes:
            If the lowest energy has already been calculated and stored 
            in `self.lowest_energy`, it will return that value directly 
            to save computation time.
        """
        
        
        # Only to be used for small instances, it is just brute force so extremely memory intensive and slow
        if self.lowest_energy is not None:
            return self.lowest_energy
        else:
            all_energies = self.get_all_energies()

        lowest_energy = np.min(all_energies)

        return lowest_energy

    def get_boltzmann_factor(
        self, state: str, beta: float = 1.0
    ) -> float:

        """ 
        Get un-normalised boltzmann probability of a given state 

        Args:
            state (str): configuration of spins for which probability is to be calculated 
            beta (float): inverse temperature (1/T) at which the probability is to be calculated.
        
        Returns:
            float corresponding to the un-normalised boltzmann probability of the given state.
        """
        E = self.get_energy(state)
        r = np.exp(-1 * beta * E, dtype = np.longdouble)

        return r

    
    def get_boltzmann_factor_from_energy(self, E, beta: float = 1.0
    ) -> float:

        """
        Get un-normalized Boltzmann probability for a given energy.

        Args:
            E (float): Energy for which the Boltzmann factor is to be calculated.
            beta (float): Inverse temperature (1/T) at which the probability is to be calculated.

        Returns:
            float: The un-normalized Boltzmann probability for the given energy.
        """
        
        return np.exp(-1 * beta * E, dtype = np.longdouble)
    
            
    @property
    def get_J(self):
        return self.J
    
    @property
    def get_h(self):
        return self.h






    
###########################################################################################
## EXACT SAMPLING on MODEL ##
###########################################################################################



class Exact_Sampling(IsingEnergyFunction):

    def __init__(self, model: IsingEnergyFunction,  beta:float= 1.0, verbose= False) -> None :

        super().__init__(model.get_J, model.get_h, model.name)
    
        self.beta = beta
        self.exact_sampling_status = False
        self.run_exact_sampling(self.beta, verbose= verbose)

    def sampling_summary(self, plot_dist:bool=True, show_threshold=0.01):
        
        if self.exact_sampling_status :
            tmp = np.array(list(self.boltzmann_pd.values()))
            
            print(tmp)
            count_non_zero = len(tmp[tmp > show_threshold])
            
            print("=============================================")
            print("     MODEL : "+str(self.name)+" |  beta : "+str(self.beta) )
            print("=============================================")
            
            
            print("Num Most Probable States : " + str( count_non_zero )   )
            print("Entropy : " + str( self.get_entropy() ))
            print("---------------------------------------------")

            if plot_dist:
                plot_bargraph_desc_order(self.boltzmann_pd, label= 'Boltzmann Dist.', plot_first_few= count_non_zero)

        else:
            raise RuntimeError("Please Run Exact Sampling at any specified temperature first")

    def get_boltzmann_distribution(
        self, beta:float = 1.0, sorted:bool = False, save_distribution:bool = False , return_dist:bool= True, plot_dist:bool = False, verbose:bool= False
    ) -> dict :
        """ Get normalised boltzmann distribution over states 

            ARGS:
            ----
            beta : inverse temperature (1/ T)
            sorted  : if True then the states are sorted in in descending order of their probability
            save_dist : if True then the boltzmann distribution is saved as an attribute of this class -> boltzmann_pd 
            plot_dist : if True then plots histogram corresponding to the boltzmann distribution

            RETURNS:
            -------
            'dict' corresponding to the distribution
        """
        all_configs = [f"{k:0{self.num_spins}b}" for k in range(0, 2 ** (self.num_spins))]
        bltzmann_probs = dict( [ ( state, self.get_boltzmann_factor(state, beta= beta) ) for state in tqdm(all_configs, desc= 'running over all possible configurations', disable= not verbose ) ] )
        
        
        #added to try allow for low temperature case, but a bit useless
        arr = bltzmann_probs.values()
        arr = np.array(list(arr))
        arr[arr == None] = np.nan
        arr[arr<1E-15] = 0
        
        partition_sum=np.sum(arr)


        
        prob_vals=arr*(1./partition_sum)

        bpd= dict(zip(all_configs, prob_vals ))
        bpd_sorted_desc= value_sorted_dict( bpd, reverse=True )
        
        
        if save_distribution :
            self.boltzmann_pd = DiscreteProbabilityDistribution(bpd_sorted_desc)

        if plot_dist:
                plt.figure(2)
                plot_bargraph_desc_order(bpd_sorted_desc, label="analytical",plot_first_few=30); plt.legend()
        
        if return_dist :   
            if sorted: 
                return bpd_sorted_desc
            else :
                return bpd


    def run_exact_sampling(self, beta:float, verbose:bool= False ) -> None :
        """ Running this function executes the 'get_boltzmann_distribution' function, thus exhaustively enumerating all possible
            configurations of the system and saving the distribution as an attribute 'boltzmann_pd'. 

            NOTE:   This saves the requirement of recalculating the analytical distribution for any of the functions depending explicitly 
                    on the analytical boltzmann distribution.
                    Run this function before calling any of the methods that uses the analytical boltzmann distribution. 
                    It is recommended not to run this for num_spins > 20, as it is highly ineffecient.

            ARGS:
            ----
            beta : inverse temperature

        """
        self.exact_sampling_status = True
        self.beta = beta
        if verbose : print("Running Exact Sampling | beta : ", beta)
        self.get_boltzmann_distribution(beta= beta, save_distribution= True, return_dist= False, verbose= verbose)
        if verbose : print("saving distribution to model ...")
        

        


    def get_observable_expectation(self, observable) -> float:
        """ Return expectation value of a classical observables

            ARGS :
            ----
            observable: Must be a function of the spin configuration which takes an 'np.array' / 'str' of binary elements as input argument and returns a 'float'
            beta: inverse temperature

        """
        # all_configs = np.array(list(itertools.product([1, 0], repeat=self.num_spins)))
        all_configs = [f"{k:0{self.num_spins}b}" for k in range(0, 2 ** (self.num_spins))]
        

        return sum(
            [
                self.boltzmann_pd[config]
                * observable(config)

                for config in all_configs
            ]
        )

    def get_entropy(self):
        tmp = sorted(np.array(list(self.boltzmann_pd.values())), reverse= True)
        entropy = 0
        for val in tmp :
            if val > 0.00001 :
                entropy += -1 * val * np.log2(val)
            else: 
                return entropy
                


    def get_kldiv(self, q: dict, beta: Union[float, None]= None) -> float :
        """ Return calculated KL-divergence of the boltzmann distribution wrt. a given distribution i.e 
            D_kl( boltzmann|| q)

            ARGS:
            ----
            q : given distribution 
            beta : inverse temperature of the model 
        
        """
        ## check current beta and exact-sampling status
        if beta == None : beta = self.beta
        elif isinstance(beta, float):
            if beta != self.beta : 
                raise ValueError("Current beta is different from model beta. Please 'run_exact_sampling' with appropriate beta value ")
                # bltz_dist = self.get_boltzmann_distribution(beta= beta)
        if self.exact_sampling_status :
            bltz_dist = self.boltzmann_pd
        else :             
            bltz_dist = self.get_boltzmann_distribution(beta= beta)
        
        
        ## check q 
        q_vals = list(q.values())
        assert np.sum(q_vals) == 1 , " given distribution is not normalised "
        all_configs = [f"{k:0{self.num_spins}b}" for k in range(0, 2 ** (self.num_spins))]
        if  set(q.keys()) !=  set(all_configs):
            raise ValueError(" given distribution is not defined over all possible configurations ") 

        ## re-order
        bltz_dist = DiscreteProbabilityDistribution(bltz_dist)
        q = DiscreteProbabilityDistribution(q)
        q.normalise()

        ## calc
        bltz_dist = bltz_dist.index_sorted_dict()
        q = q.index_sorted_dict()

        p = list(bltz_dist.values())
        q = list(q.values())

        return sum(p[i] * np.log2(p[i]/q[i]) for i in range(len(p)) if p[i]!=0)

    def get_jsdiv(self, q, beta: Union[float, None]= None) -> float :
        """ Return calculated KL-divergence of the boltzmann distribution wrt. a given distribution i.e 
            D_js( boltzmann ,  q)

            ARGS:
            ----
            q : given distribution 
            beta : inverse temperature of the model 
        
        """        
        ## check current beta and exact-sampling status
        if beta == None : beta = self.beta
        elif isinstance(beta, float):
            if beta != self.beta : 
                raise ValueError("Current beta is different from model beta. Please 'run_exact_sampling' with appropriate beta value ")
                # bltz_dist = self.get_boltzmann_distribution(beta= beta)
        if self.exact_sampling_status :
            bltz_dist = self.boltzmann_pd
        else :             
            bltz_dist = self.get_boltzmann_distribution(beta= beta)
        
        
        ## checks
        q_vals = list(q.values())
        assert np.sum(q_vals) == 1 , " given distribution is not normalised "
        all_configs = [f"{k:0{self.num_spins}b}" for k in range(0, 2 ** (self.num_spins))]
        assert set(q.keys()).issubset(all_configs) , " given distribution is not defined over all possible configurations " 
        
        ## create mixed distribution
        m = {}
        for key in bltz_dist.keys():
            m[key] = 0.5 * ( bltz_dist[key] + q[key] ) 
        
        return 0.5 * self.get_kldiv(bltz_dist, m) + 0.5 * self.get_kldiv(q, m)





