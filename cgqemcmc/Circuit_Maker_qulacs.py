import numpy as np
from qulacs import QuantumState, QuantumCircuit
from scipy.linalg import expm
from qulacs.gate import DenseMatrix
from qulacs.gate import X, Y, Z  , Pauli, Identity, merge
from itertools import combinations





class Circuit_Maker:

    """
    Class which initialises a circuit for a given problem.

    Can be initialised, then itteritively tasked with making a circuit given a new input string s.

    Taken mostly from https://github.com/pafloxy/quMCMC but restructured

    """

    def __init__(self, model, gamma, time, single_qubit_mixer=True):

        """
        Initialise Circuit_Maker object.

        ARGS:

        model: obj
            energy model object, normally IsingEnergyFunction from energy_models

        gamma: float
            parameter in QeMCMC circuit

        time: float
            parameter in QeMCMC circuit

        single_qubit_mixer: bool
            inherited from QuMCMC, honestly not 100% sure what it does.
            #I shouldr eally check this





        """

        self.single_qubit_mixer = single_qubit_mixer
        self.time = time
        self.gamma = gamma
        self.model = model
        self.h = self.model.get_h
        self.J = self.model.get_J
        self.delta_time = 0.8
        self.n_spins = model.num_spins   
        self.alpha = model.alpha
        self.pauli_index_list:list=[1,1] #cant really remember what this does
        self.num_trotter_steps = int(np.floor((self.time / self.delta_time)))

        #create trotter circuit that is irrelevant of the input string
        self.qc_evol_h1 = self.fn_qc_h1()  
        self.qc_evol_h2 = self.fn_qc_h2()
        self.trotter_ckt = self.trottered_qc_for_transition(self.qc_evol_h1, self.qc_evol_h2,self.num_trotter_steps)


        # init_qc=initialise_qc(n_spins=n_spins, bitstring='1'*n_spins)

    def build_circuit(self, s:str):
        #build a circuit for a given bitstring
        qc_s = self.initialise_qc(bitstring= s)
        qc_for_s = self.combine_2_qc(qc_s, self.trotter_ckt)# i can get rid of this!

        return qc_for_s

    def get_state_obtained_binary(self, s):
        #get the output bitstring s' given s

        qc_for_s = self.build_circuit(s)

        q_state= QuantumState(qubit_count=self.n_spins)
        q_state.set_zero_state()
        qc_for_s.update_quantum_state(q_state)

        state_obtained=q_state.sampling(sampling_count=1)[0]
        state_obtained_binary=f"{state_obtained:0{self.n_spins}b}"
        return state_obtained_binary
    
    def get_state_obtained_probs(self,s):
        #get the probability of each s' given s

        qc_for_s = self.build_circuit(s)

        q_state = QuantumState(qubit_count = self.n_spins )
        q_state.set_zero_state()
        qc_for_s.update_quantum_state(q_state)

        probs = np.absolute((q_state.get_vector()))**2

        return probs






    def initialise_qc(self,bitstring) -> QuantumCircuit :
        """
        Initialises a quantum circuit with n_spins number of qubits in a state defined by "bitstring"
        """
        qc_in=QuantumCircuit(qubit_count=self.n_spins)
        len_str_in = len(bitstring)
        assert len_str_in==qc_in.get_qubit_count(), "len(bitstring) should be equal to number_of_qubits/spins"

        for i in range(0,len(bitstring)):
            if bitstring[i]=="1":
                qc_in.add_X_gate(len_str_in - 1 - i)
        return qc_in



    def fn_qc_h1(self) -> QuantumCircuit :
        """
        Create a Quantum Circuit for time-evolution under
        hamiltonain H1 (described in the paper)
        H1= -(1-gamma)*alpha*sum_{j=1}^{n}[(h_j*Z_j)] + gamma *sum_{j=1}^{n}[(X_j)]

        
        """
        a=self.gamma
        b_list = ((self.gamma-1)*self.alpha)* np.array(self.h)
        qc_h1 = QuantumCircuit(self.n_spins)
        if self.single_qubit_mixer:
            for j in range(0, self.n_spins):
                # unitary_gate=DenseMatrix(index=num_spins-1-j,
                #                 matrix=np.round(expm(-1j*delta_time*(a*X(2).get_matrix()+b_list[j]*Z(2).get_matrix())),decimals=6)
                #                 )# this will change accordingly.

                Matrix = np.round(expm(-1j*self.delta_time*(a*X(2).get_matrix()+b_list[j]*Z(2).get_matrix())),decimals=6)

                unitary_gate=DenseMatrix(index=self.n_spins-1-j,
                                matrix = Matrix)# this will change accordingly.
                qc_h1.add_gate(unitary_gate)
        else:# added by Neel

            for j in range(0,self.n_spins):

                matrix_to_exponentiate=b_list[j]*Z(2).get_matrix()
                unitary_gate=DenseMatrix(index=self.n_spins-1-j,
                                        matrix=np.round(expm(-1j*self.delta_time*matrix_to_exponentiate),decimals=6)
                                        )
                qc_h1.add_gate(unitary_gate)


        return qc_h1




    def fn_qc_h2(self) -> QuantumCircuit :
        """
        Create a Quantum Circuit for time-evolution under
        hamiltonain H2 (described in the paper)
        ARGS:
        ----
        
        """

        
        self.n_spins=np.shape(self.J)[0]
        qc_for_evol_h2=QuantumCircuit(self.n_spins)
        # calculating theta_jk
        upper_triag_without_diag=np.triu(self.J,k=1)
        theta_array=(-2*(1-self.gamma)*self.alpha*self.delta_time)*upper_triag_without_diag
        #theta_array=(-2*(1-self.gamma)*self.alpha*self.delta_time)*self.J
        pauli_z_index=[3,3]## Z tensor Z
        for j in range(0,self.n_spins-1):
            for k in range(j+1,self.n_spins):
            #for k in range(j + 1, j+2):
                #print("j,k is:",(j,k))
                target_list=[self.n_spins-1-j,self.n_spins-1-k]#num_spins-1-j,num_spins-1-(j+1)
                angle = theta_array[j,k]

                qc_for_evol_h2.add_multi_Pauli_rotation_gate(index_list=target_list,pauli_ids=pauli_z_index,angle = angle)
                
                
        if not(self.single_qubit_mixer):
            pauli_x_index=self.pauli_index_list
            indices=list(range(0,self.n_spins))
            r=len(pauli_x_index)
            all_poss_combn_asc_order=list(combinations(indices,r))
            for i in range(0,len(all_poss_combn_asc_order)):
                target_list=list(all_poss_combn_asc_order[i])
                angle = -1 * self.gamma * self.delta_time ## @pafloxy : make the angle 'gamma' dependent

                qc_for_evol_h2.add_multi_Pauli_rotation_gate(index_list=target_list,
                                                                pauli_ids=pauli_x_index,
                                                                angle=angle)

            #print("qc_for_evol_h2")
            #circuit_drawer(qc_for_evol_h2, "mpl")
            #plt.show()
        return qc_for_evol_h2


    def trottered_qc_for_transition(self, qc_h1: QuantumCircuit, qc_h2: QuantumCircuit, num_trotter_steps: int) -> QuantumCircuit:
        """ Returns a trotter circuit (evolution_under_h2 X evolution_under_h1)^(r-1) (evolution under h1)"""
        qc_combine=QuantumCircuit(self.n_spins)
        for _ in range(0,num_trotter_steps-1):

            qc_combine.merge_circuit(qc_h1)
            qc_combine.merge_circuit(qc_h2)

        qc_combine.merge_circuit(qc_h1)

        #print("qc_combine")
        #circuit_drawer(qc_combine, "mpl")
        #plt.show()
        return qc_combine


    def combine_2_qc(self, init_qc: QuantumCircuit, trottered_qc: QuantumCircuit) -> QuantumCircuit:
        """ Function to combine 2 quantum ckts of compatible size.
            In this project, it is used to combine initialised quantum ckt and quant ckt meant for time evolution
        """
        qc_merge=QuantumCircuit(self.n_spins)
        qc_merge.merge_circuit(init_qc)
        qc_merge.merge_circuit(trottered_qc)
        return qc_merge


