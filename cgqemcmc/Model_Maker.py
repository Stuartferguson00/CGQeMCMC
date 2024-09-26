import numpy as np
from .energy_models import IsingEnergyFunction
import itertools



class Model_Maker:
    # Class to control the initialisation of an energy model. 
    # It might seem a bit convoluted, but will allow for more complex models to be made in future.
    def __init__(self, n_spins:int, model_type:str, name:str, J: np.ndarray = None, h: np.ndarray = None, cost_function_signs: list = [-1,-1]):
        self.name = name
        self.n_spins = n_spins
        self.cost_function_signs = cost_function_signs
        if type(model_type) is not str:
            print("model type must be a string representing the model you request")
        elif model_type == "Fully Connected Ising":
            self.make_fully_connected_Ising() 
        elif model_type == "1D Ising":
            self.make_1D_Ising() 
        elif model_type == "input_J":
            self.J = J
            self.h = h
            self.model = IsingEnergyFunction(self.J, self.h, name=self.name, cost_function_signs = self.cost_function_signs)

    def make_fully_connected_Ising(self):
        shape_of_J = (self.n_spins, self.n_spins)
        J = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
        J_tril = np.tril(J, -1)
        J_triu = J_tril.transpose()
        J = J_tril + J_triu
        
        h = np.round(np.random.normal(0, 1, self.n_spins), decimals=4)

        self.model = IsingEnergyFunction(J, h, name=self.name, cost_function_signs = self.cost_function_signs)
        
    def make_1D_Ising(self):
        print("I have not analysed 1d ising models, so double check function 'make_1D_Ising' before using")
        h = np.round(np.random.normal(0, 1, self.n_spins), decimals=4)
        shape_of_J = (self.n_spins, self.n_spins)
        J = np.zeros(shape_of_J)
        
        
        J_rand = np.round(np.random.normal(0, 1, shape_of_J), decimals=4)
        J_tril = np.tril(J_rand, -1)
        J_triu = J_tril.transpose()
        J_rand = J_tril + J_triu



        #loop throgh and find the difference in bitstrings.
        #When the ith bitstring is different (by a value of 1) from the jth bitstring add 1 to Q[i,j]
        for i in range(self.n_spins):
            for j in range(self.n_spins):
                if abs(i-j) == 1 or abs(i-j) == self.n_spins-1:
                    J[i, j] = 1

        J = J * J_rand
        self.model = IsingEnergyFunction(J, h, name=self.name, cost_function_signs = self.cost_function_signs)
        
        
