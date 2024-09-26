



try:
    from qulacs import QuantumStateGpu as QuantumStateGpu
    from qulacs import QuantumState
    print("Using GPU qulacs")
except:
    from qulacs import QuantumState
    print("Using CPU qulacs as qulacs install is not configured for GPU simulation ")
    
from qulacs.gate import DenseMatrix
from qulacs.gate import X, Y, Z  , Pauli, Identity, merge
from qulacs import QuantumCircuit

from scipy.linalg import expm
import numpy as np
import torch
from cgqemcmc.energy_models import IsingEnergyFunction
from typing import Union



class Circuit_Maker:

    """
    Class which initialises a circuit for a given problem.

    Can be initialised, then tasked with making a circuit given a new input string s.

    Taken mostly from https://github.com/pafloxy/quMCMC but restructured

    """

    def __init__(self, model: IsingEnergyFunction, gamma: Union[float, tuple] , time: Union[int, tuple]):

        """
        Initialise Circuit_Maker object.

        Args:

        model: obj
            energy model object, normally IsingEnergyFunction from energy_models
        gamma: float or tuple
            parameter in QeMCMC circuit
        time: int or tuple
            parameter in QeMCMC circuit






        """

        self.time = time
        self.gamma = gamma
        self.model = model
        
        # To fix sign issues
        # Very helpful when using different forms of Ising models
        if self.model.cost_function_signs[0] == 1:
            self.h = -self.model.h
        else:
            self.h = self.model.h
        if self.model.cost_function_signs[1] == 1:
            self.J = -self.model.J
        else:
            self.J = self.model.J
            
            
        self.delta_time = 0.8
        self.n_spins = model.num_spins   
        self.alpha = model.alpha
        self.pauli_index_list:list=[1,1] #cant really remember what this does
        self.num_trotter_steps = int(np.floor((self.time / self.delta_time)))

        #create trotter circuit that is irrelevant of the input string
        self.qc_evol_h1 = self.fn_qc_h1()  
        self.qc_evol_h2 = self.fn_qc_h2()
        self.trotter_ckt = self.trottered_qc_for_transition(self.qc_evol_h1, self.qc_evol_h2, self.num_trotter_steps)


        # init_qc=initialise_qc(n_spins=n_spins, bitstring='1'*n_spins)

    def build_circuit(self, s:str) -> QuantumCircuit:
        """
        Build a quantum circuit for a given bitstring.
        Args:
            s (str): The bitstring for which the quantum circuit is to be built.
        Returns:
            QuantumCircuit: The combined quantum circuit for the given bitstring.
        """
        #build a circuit for a given bitstring
        qc_s = self.initialise_qc(s)
        qc_for_s = self.combine_2_qc(qc_s, self.trotter_ckt)# i can get rid of this!

        return qc_for_s

    def get_state_obtained_binary(self, s: str) -> str:
        """
        Get the output bitstring s' given an input bitstring s.
        This method builds a quantum circuit based on the input bitstring `s`, 
        initializes a quantum state (using GPU if available), updates the quantum 
        state with the circuit, and then samples the resulting state to obtain the 
        output bitstring in binary format.
        Args:
            s (str): The input bitstring.
        Returns:
            str: The output bitstring in binary format.
        """
        #get the output bitstring s' given s

        qc_for_s = self.build_circuit(s)
        num_gpus = torch.cuda.device_count()
        if num_gpus !=0:
            q_state= QuantumStateGpu(qubit_count=self.n_spins)
        else:
            q_state= QuantumState(qubit_count=self.n_spins)
        q_state.set_zero_state()
        qc_for_s.update_quantum_state(q_state)

        state_obtained=q_state.sampling(sampling_count=1)[0]
        state_obtained_binary=f"{state_obtained:0{self.n_spins}b}"
        return state_obtained_binary


    def initialise_qc(self,s : str) -> QuantumCircuit :
        """
        Initializes a quantum circuit based on a given binary string.
        Args:
            s (str): A binary string where each character represents the initial state of a qubit.
                    '1' indicates that the qubit should be in the |1⟩ state, and '0' indicates that
                    the qubit should be in the |0⟩ state.
        Returns:
            QuantumCircuit: A quantum circuit with the specified initial states for each qubit.
        Raises:
            AssertionError: If the length of the input string `s` does not match the number of qubits.
        """
 
        qc_in=QuantumCircuit(qubit_count=self.n_spins)
        len_str_in = len(s)
        assert len_str_in==qc_in.get_qubit_count(), "len(s) should be equal to number_of_qubits/spins"

        for i in range(0,len(s)):
            if s[i]=="1":
                qc_in.add_X_gate(len_str_in - 1 - i)
        return qc_in



    def fn_qc_h1(self) -> QuantumCircuit :
        """
        Create a Quantum Circuit for time-evolution under Hamiltonian H1.
        The Hamiltonian H1 is described as:
        H1 = -(1-gamma)*alpha*sum_{j=1}^{n}[(h_j*Z_j)] + gamma *sum_{j=1}^{n}[(X_j)]
        This function constructs a quantum circuit that simulates the time evolution
        under the Hamiltonian H1 for a given time step `delta_time`.
        Returns:
            QuantumCircuit: A quantum circuit representing the time evolution under H1.
        """

        a=self.gamma
        b_list = ((self.gamma-1)*self.alpha)* np.array(self.h)
        qc_h1 = QuantumCircuit(self.n_spins)
        for j in range(0, self.n_spins):

            Matrix = np.round(expm(-1j*self.delta_time*(a*X(2).get_matrix()+b_list[j]*Z(2).get_matrix())),decimals=6)

            unitary_gate=DenseMatrix(index=self.n_spins-1-j,
                            matrix = Matrix)
            qc_h1.add_gate(unitary_gate)

        return qc_h1




    def fn_qc_h2(self) -> QuantumCircuit :
        
        """
        Hamiltonian H2 (described in the paper).
        This function constructs a quantum circuit that simulates the 
        time evolution of a system under the Hamiltonian H2. The Hamiltonian 
        is represented by the interaction matrix `self.J`, and the evolution 
        parameters are determined by `self.gamma` and `self.delta_time`.
        Returns:
        -------
        QuantumCircuit
            A quantum circuit representing the time evolution under the Hamiltonian H2.
        """

        

        
        self.n_spins=np.shape(self.J)[0]
        qc_for_evol_h2=QuantumCircuit(self.n_spins)
        upper_triag_without_diag=np.triu(self.J,k=1)
        theta_array=(-2*(1-self.gamma)*self.alpha*self.delta_time)*upper_triag_without_diag
        pauli_z_index=[3,3]# ZZ
        for j in range(0,self.n_spins-1):
            for k in range(j+1,self.n_spins):

                target_list=[self.n_spins-1-j,self.n_spins-1-k]
                angle = theta_array[j,k]

                qc_for_evol_h2.add_multi_Pauli_rotation_gate(index_list=target_list,pauli_ids=pauli_z_index,angle = angle)
                

        return qc_for_evol_h2


    def trottered_qc_for_transition(self, qc_h1: QuantumCircuit, qc_h2: QuantumCircuit, num_trotter_steps: int) -> QuantumCircuit:
        """
        Returns a Trotterized quantum circuit.
        This method constructs a quantum circuit that approximates the evolution under the combined Hamiltonians
        H1 and H2 using the Trotter-Suzuki decomposition. The resulting circuit is of the form:
        (evolution_under_h2 X evolution_under_h1)^(num_trotter_steps-1) (evolution under h1).
        Args:
            qc_h1 (QuantumCircuit): The quantum circuit representing the evolution under Hamiltonian H1.
            qc_h2 (QuantumCircuit): The quantum circuit representing the evolution under Hamiltonian H2.
            num_trotter_steps (int): The number of Trotter steps to use in the decomposition.
        Returns:
            QuantumCircuit: The Trotterized quantum circuit representing the combined evolution.
        """
        
        
        qc_combine=QuantumCircuit(self.n_spins)
        for _ in range(0,num_trotter_steps-1):

            qc_combine.merge_circuit(qc_h1)
            qc_combine.merge_circuit(qc_h2)

        qc_combine.merge_circuit(qc_h1)

        return qc_combine


    def combine_2_qc(self, init_qc: QuantumCircuit, trottered_qc: QuantumCircuit) -> QuantumCircuit:
        """ 
            Function to combine 2 quantum circuits of compatible size.
            
        """
        qc_merge=QuantumCircuit(self.n_spins)
        qc_merge.merge_circuit(init_qc)
        qc_merge.merge_circuit(trottered_qc)
        return qc_merge


