##########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
from typing import Optional
from tqdm import tqdm
import random
from cgqemcmc.basic_utils import MCMCChain, MCMCState
from cgqemcmc.energy_models import IsingEnergyFunction
import warnings






class MCMC_Classical:
    """
    Class MCMC_Classical:
        A class to perform Markov Chain Monte Carlo (MCMC) simulations for the Ising model.
        
    Methods:
        __init__(self, model: IsingEnergyFunction, temp: float, method: str = "uniform")
            Initialize the MCMC routine for the Ising model.
        update_uniform(self, current_state_bitstring: str) -> str
            Performs a uniform update on the current state bitstring.
        get_random_state(self, num_spins: int) -> str
            Generate a random state for a given number of spins.
        update_local(self, current_state_bitstring: str) -> str
            Performs a local update on the current state bitstring.
        run(self, n_hops: int, initial_state: Optional[str] = None, name: str = "classical MCMC", verbose: bool = False, sample_frequency: int = 1)
            Run the MCMC simulation for a specified number of hops.
            
        """
    
    
    def __init__(self, model: IsingEnergyFunction , temp, method = "uniform"):
        """
        Initialize the MCMC routine for the Ising model.
        
        Args:
        model (IsingEnergyFunction): The energy function of the Ising model.
        temp (float): The temperature of the system.
        method (str, optional): The update method to use. Options are "uniform" or "local". Default is "uniform".
        
        Attributes:
        model (IsingEnergyFunction): The energy function of the Ising model.
        temp (float): The temperature of the system.
        beta (float): The inverse temperature (1/temp).
        n_spins (int): The number of spins in the model.
        method (str): The update method to use.
        update (function): The update function corresponding to the chosen method.
        """


        self.model = model
        self.temp = temp
        self.beta = 1/self.temp
        self.n_spins = model.num_spins
        self.method = method

        if self.method == "uniform":
            self.update = self.update_uniform
        elif self.method == "local":
            self.update = self.update_local
        else:
            print("method name is incorrect. Choose from: 'uniform' or 'local'")
                
            
        

    def update_uniform(self,current_state_bitstring:str) -> str:
        """
        Updates the current state bitstring by generating a new random state bitstring of the same length.
        Args:
            current_state_bitstring (str): The current state represented as a bitstring.
        Returns:
            str: A new random state bitstring of the same length as the input.
        """


        s_prime = get_random_state(len(current_state_bitstring))

        return s_prime
    
    

    

    
    def update_local(self,current_state_bitstring:str)-> str:
        """
        Update the local state by flipping a randomly chosen spin in the current state bitstring.
        Args:
            current_state_bitstring (str): The current state represented as a bitstring.
        Returns:
            str: The new state bitstring after flipping a randomly chosen spin.
        """

        
        # Randomly choose which spin to flip
        choice = np.random.randint(0,self.n_spins)

        # Flip the chosen spin
        c_s = list(current_state_bitstring)
        c_s[choice] = str(int(c_s[choice]) ^ 1)
        
        # Return the new state as a bitstring
        s_prime = ''.join(c_s)   
        return s_prime

        
    




    def run(self,
        n_hops: int,
        initial_state: Optional[str] = None,
        name:str = "classical MCMC",
        verbose:bool = False,
        sample_frequency:int = 1):



        """
        Run the classical MCMC algorithm for a specified number of hops.
        Parameters:
        n_hops : int
            The number of hops (iterations) to perform in the MCMC algorithm.
        initial_state : Optional[str], optional
            The initial state to start the MCMC algorithm. If None, a random state is generated. Default is None.
        name : str, optional
            The name of the MCMC run. Default is "classical MCMC".
        verbose : bool, optional
            If True, prints detailed information during the run. Default is False.
        sample_frequency : int, optional
            The frequency at which to sample and store states in the MCMC chain. Default is 1.
        Returns:
        MCMCChain
            The MCMC chain containing the sequence of states visited during the run.
        """








        if name is None:
            name = self.method + " MCMC"

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


        #define MCMC chain
        mcmc_chain = MCMCChain([current_state], name= name)
        
        
        # Do MCMC 
        for i in tqdm(range(0, n_hops), desc='Run '+name, disable= not verbose ):

            # Propose a new state
            s_prime = self.update(current_state.bitstring)
            
            # Find energy of the new state
            energy_sprime = self.model.get_energy(s_prime)
            
            # Decide whether to accept the new state
            accepted = test_accept(energy_s, energy_sprime, temperature=self.temp)
            
            # If accepted, update current_state
            if accepted:
                current_state = MCMCState(s_prime, accepted, energy_s, pos = i)
                energy_s = energy_sprime
                
                
            # if time to sample, add state to chain
            if i//sample_frequency == i/sample_frequency and i != 0 :
                mcmc_chain.add_state(MCMCState(current_state.bitstring, accepted, energy_s, pos = i))

        return mcmc_chain





    def test_probs(self, energy_s: float, energy_sprime: float) -> float:
        
        """
        Calculate the probability ratio between two states based on their energies.
        This function computes the exponential factor used in the Metropolis-Hastings 
        algorithm to determine the acceptance probability of a new state s' given 
        the current state s. The probability ratio is calculated as exp(-(E(s') - E(s)) / T),
        where E(s) and E(s') are the energies of the current and proposed states, respectively,
        and T is the temperature.
        Args:
            energy_s (float): The energy of the current state s.
            energy_sprime (float): The energy of the proposed state s'.
        Returns:
            float: The probability ratio exp(-(E(s') - E(s)) / T).
        """

        delta_energy = energy_sprime - energy_s  # E(s')-E(s)
        exp_factor = np.exp(-delta_energy / self.temp)

        return exp_factor


def test_accept(
    energy_s: float, energy_sprime: float, temperature: float = 1.
) -> MCMCState:
    """
    Accepts the state "sprime" with probability A ( i.e. min(1,exp(-(E(s')-E(s))/ temp) )
    and s_init with probability 1-A.
    """
    delta_energy = energy_sprime - energy_s  # E(s')-E(s)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            exp_factor = np.exp(-delta_energy / temperature)
        except RuntimeWarning as e:
            
            exp_factor = 0
            print("Error in exponantial: delta_energy = ", delta_energy, "temperature = ", temperature, " energy_s = ", energy_s, " energy_sprime = ", energy_sprime)
            
    acceptance = min(
        1, exp_factor
    )  # for both QC case as well as uniform random strategy, the transition matrix Pij is symmetric!

    return acceptance > np.random.rand()

    
def get_random_state(num_spins: int) -> str:
    """
    Generate a random state for a given number of spins.
    Args:
        num_spins (int): The number of spins in the system.
    Returns:
        str: A bitstring representing the random state.
    """

    # Define the size of state space
    state_space = 2 ** (num_spins)
    
    # Generate a random state
    next_state = random.randrange(0,state_space,1)

    # Convert the state from integer to a bitstring
    s_prime = f"{next_state:0{num_spins}b}"
    return s_prime





