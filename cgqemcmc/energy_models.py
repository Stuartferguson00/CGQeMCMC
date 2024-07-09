###########################################################################################
## IMPORTS ##
###########################################################################################
import numpy as np
import matplotlib.pyplot as plt
from .prob_dist import  value_sorted_dict, DiscreteProbabilityDistribution
from .basic_utils import plot_bargraph_desc_order
from typing import Union
from tqdm import tqdm
import seaborn as sns

from qulacs import PauliOperator, Observable
import itertools

###########################################################################################
## ENERGY MODEL ##
###########################################################################################

class IsingEnergyFunction:
    """ 
    A class to build the Ising Energy Function from self  
    
    Heavily modified from https://github.com/pafloxy/quMCMC to add functionality
    
    
    """

    def __init__(self, J: np.array, h: np.array, name:str = None, negative_energy:bool = True) -> None:
        """
            ARGS:
            ----
            J: weight-matrix of the interactions between the spins 
            h: local field to the spins 
            name: Name of ising model

        """
        self.negative_energy = negative_energy
        self.J = J
        self.h = h
        self.S = None
        self.all_energies = None
        self.lowest_energy  = None
        self.num_spins = len(h)
        self.alpha = np.sqrt(self.num_spins) / np.sqrt( sum([J[i][j]**2 for i in range(self.num_spins) for j in range(i)]) + sum([h[j]**2 for j in range(self.num_spins)])  )
        if name is None: 
            self.name = 'JK_random'
        else : self.name = name 
        
        #10 optional optional states to use for fair starting state between different simulations
        
        self.initial_state = []
        for i in range(10): 
            self.initial_state.append(np.random.randint(0, 2, self.num_spins, dtype = int))

        
    def change_J(self, J):
        #added so I can change J post-initialisation 
        self.J = J
        self.alpha = np.sqrt(self.num_spins) / np.sqrt( sum([self.J[i][j]**2 for i in range(self.num_spins) for j in range(i)]) + sum([self.h[j]**2 for j in range(self.num_spins)])  )

        
    
    @property
    def get_J(self):
        return self.J
    
    @property
    def get_h(self):
        return self.h

    # @property
    def model_summary(self, plot= True) :
        
        print("=============================================")
        print("            MODEL : "+str(self.name) )
        print("=============================================")
        
        
        print("Non-zero Interactions (J) : "+ str( int(np.count_nonzero(self.J) /2)) + ' / '+str( int( 0.5 * self.num_spins * (self.num_spins - 1))) )
        print("Non-zero Bias (h) : "+ str( int(np.count_nonzero(self.h) )) + ' / ' + str( self.num_spins ) )
        print("---------------------------------------------")

        print("Average Interaction Strength <|J|> : ", np.mean(np.abs(self.J)))
        print("Average Bias Strength <|h|>: ", np.mean(np.abs(self.h)))
        print("alpha : ", self.alpha )
    
        print("---------------------------------------------")


        # sns.set()
        if plot:
            plt.figure(figsize=(16,10))
            sns.heatmap(self.J, square= True, annot= False, cbar= True)
            
    # def get_hamiltonian
    def get_hamiltonian(self):

        J = self.get_J; h = self.get_h

        hamiltonian = Observable(self.num_spins)

        for i in range (0, self.num_spins):

            pstr = 'Z ' + str(i)
            hamiltonian.add_operator(PauliOperator(pstr, coef= h[i]))

            for j in range(0, i):

                pstr = 'Z ' + str(i) + ' ' + 'Z ' + str(j)
                hamiltonian.add_operator(PauliOperator(pstr, coef= J[i,j]))

        return hamiltonian
    
    def get_energy(self, state: Union[str, np.array]) -> float:
        """ Returns the energy of a given state

            ARGS:
            ----
            state : configuration of spins for which the energy is required to be calculated.
                    NOTE:  if input is an numpy array then it should only consist of bipolar values -> {+1, -1}

        """
        if self.all_energies is None:    
            energy = self.calc_an_energy(state)
        
        else:
            
            #if not isinstance(state, str):
            #    state = np.array([-1 if elem == "-1" else 1 for elem in state])
            try:
                energy = self.all_energies[int(state,2)]   
            except:        
                state = ''.join(str(s) for s in state)
                energy = self.all_energies[int(state,2)]
        return energy
            
    def calc_an_energy(self,state):

        
        if isinstance(state, list) or isinstance(state, type(np.array(0))):
            
            state = np.array(state, dtype = int)
            state = "".join(str(s) for s in state)
            if "-1" in state:
                print("Oh you fucked it 1")
            state = np.array([-1 if elem == "0" else 1 for elem in state])
        elif isinstance(state, str):
            if "-1" in state:
                print("Oh you fucked it 2")
            state = np.array([-1 if elem == "0" else 1 for elem in state])
        #energy = 0.5 * np.dot(state.transpose(), self.J.dot(state)) + np.dot(
        #            self.h.transpose(), state)
        try:
            #THIS ONLY WORKS IF THE INPUT IS NOT UPPER DIAGONAL.
            energy = 0.5 * np.dot(state.transpose(), self.J.dot(state)) + np.dot(
                        self.h.transpose(), state)
            if self.negative_energy:
                energy = - energy
        except:
            print("the weird error again. ")
            print("state: ")
            print(state)
            energy = 1000

        return energy
        
    def get_all_energies(self):
        self.S = [''.join(i) for i in itertools.product('01', repeat=self.num_spins)]
        self.all_energies = np.zeros(len(self.S))
        for state in self.S:

            self.all_energies[int(state,2)] = self.calc_an_energy(state)
            
            
    def get_lowest_energies(self,num_states):
        #only to be used for small instances, it is just brute force so extremely memory intensive and slow
        
        if self.all_energies is None:    
            self.get_all_energies()

        #very slow (sorts whole array)  
        self.lowest_energies, self.lowest_energy_degeneracy = self.find_lowest_values(self.all_energies, num_values=num_states)

        return self.lowest_energies, self.lowest_energy_degeneracy
    
    def find_lowest_values(self, arr, num_values=5):
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
        # Only to be used for small instances, it is just brute force so extremely memory intensive and slow
        if self.lowest_energy is not None:
            return self.lowest_energy
        else:
            if self.all_energies is None:
                self.get_all_energies()


        lowest_energy = np.min(self.all_energies)

        return lowest_energy

    def get_boltzmann_factor(
        self, state: Union[str, np.array], beta: float = 1.0
    ) -> float:

        """ Get un-normalised boltzmann probability of a given state 

            ARGS:
            ----
            state : configuration of spins for which probability is to be calculated 
            beta : inverse temperature (1/T) at which the probability is to be calculated.
        
        """
        E = self.get_energy(state)
        #print("starts")
        #print(E)
        
            
        
        r = np.exp(-1 * beta * E, dtype = np.longdouble)
        #print(r)
        return r
        
            
    
    def get_boltzmann_factor_from_energy(self, E, beta: float = 1.0
    ) -> float:
        return np.exp(-1 * beta * E, dtype = np.longdouble)
    
    from typing import List
    def _update_J(self, new_param:float, index: Union[tuple, List]):

        assert len(index) == 2
        self.J[index[0], index[1]] = new_param
        self.J[index[1], index[0]] = new_param
    
    def _update_h(self, new_param: float, index: int):

        self.h[index] = new_param





    
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





