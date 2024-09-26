import unittest
import cgqemcmc
import numpy as np
from cgqemcmc.classical_mcmc_routines import MCMC_Classical
from cgqemcmc.qulacs_CGQeMCMC import MCMC_qulacs
from cgqemcmc.Model_Maker import Model_Maker
from cgqemcmc.classical_mcmc_routines import get_random_state, test_accept

class TestCGQeMCMC(unittest.TestCase):
    """
    
    A class to test the cgqemcmc package.
    
    I can't test the CGQeMCMC very well as it is probabilitstic and complex. I can test the classical MCMC routines etc. accurately however.

    Args:
        unittest (_type_): _description_
    """
    
    def setUp(self):
        model = Model_Maker(5, "Fully Connected Ising", str(5) +" test case").model
        self.quant_MCMC = MCMC_qulacs(model, 0.5, 1, 1, 5, 1, False)
        self.class_MCMC = MCMC_Classical(model, 1, "uniform")
    
    def test_model_maker(self):
        rand_int = np.random.randint(3, 10)
        J = np.random.rand(rand_int, rand_int)
        h = np.random.rand(rand_int)
        model_ = Model_Maker(rand_int, "input_J", str(rand_int)+" test case", J, h, [-1,-1]).model

        self.assertEqual(J.tolist(), model_.J.tolist(), "J matrix not correctly mapping to model.J")
        self.assertEqual(h.tolist(), model_.h.tolist(), "h matrix not correctly mapping to model.h")
        
        J_0 = np.ones((rand_int, rand_int)) *0.00001
        h_0 = np.ones(rand_int)
        
        model_0 = Model_Maker(rand_int, "input_J", str(rand_int)+" test case", J_0, h_0, [-1,-1]).model
        energy_a = model_0.get_energy("0"*rand_int)
        energy_b = model_0.get_energy("1"*rand_int)
        
        self.assertAlmostEqual(energy_a, -energy_b, 2,"Ising model gives wrong energies for J = 0 and 0000.., 11111... energies?")
        

        
    def test_update_local(self):
        # Test case that update_local changes only one spin in the bitstring
        bitstring = ''.join(map(str, np.random.choice([0, 1], size=5)))
        
        # Call the update_local function
        new_state = self.class_MCMC.update_local(bitstring)
        
        # Calculate the Hamming distance between new_state and bitstring
        hamming_distance = sum(el1 != el2 for el1, el2 in zip(new_state, bitstring))
        
        # Assert that the Hamming distance is 1
        self.assertEqual(hamming_distance, 1, "The local update is not flipping only one spin")
        
        all_indexes = [0,1,2,3,4]
        success = 0
        # Loop to test the overall statistics of spin flips
        for i in range(2**5):
            new_state = self.class_MCMC.update_local(bitstring)
            
            index_of_difference = [i for i in range(len(bitstring)) if bitstring[i] != new_state[i]][0]

            if index_of_difference in all_indexes:
                all_indexes.remove(index_of_difference)

            if len(all_indexes) == 0:
                success = 1
                break
            
        self.assertEqual(success, 1, "The local update is not flipping all the spins with equal probability")
        
    def test_update_uniform(self):
        # Test case that update_local changes only one spin in the bitstring
        bitstring = "00000"
        
        tot_hamming_distance = 0
        num_samples = 1000000
        for i in range(num_samples):
            # Call the update_local function
            new_state = self.class_MCMC.update_uniform(bitstring)
            
            # Calculate the Hamming distance between new_state and bitstring
            tot_hamming_distance += sum(el1 != el2 for el1, el2 in zip(new_state, bitstring))
        
        avg_hamming_distance = tot_hamming_distance/(num_samples)
        self.assertAlmostEqual(avg_hamming_distance, 5/2, places=2)
        
        

    def test_test_probs(self):


        self.assertGreaterEqual(self.class_MCMC.test_probs(1.3, 0.22), 1, "The MH acceptance probability is not greater than 1 for a transition to a lower energy state")
        self.assertLess(self.class_MCMC.test_probs(0, 1), 1, "The MH acceptance probability is not less than 1 for a transition to a higher energy state")
        self.assertAlmostEqual(self.class_MCMC.test_probs(0.5,0.5),1, places=1, msg="The MH acceptance probability is not 1 for a transition to a state with the same energy")

    def test_accept(self):
        self.assertTrue(test_accept(1, 0.5), "The accept function is not accepting a transition to a lower energy state")
        
        self.assertTrue(test_accept(0.1,0.1), "The accept function is not accepting a transition to the same state")
        
        accepted = 0
        num_samples =1000
        for i in range(num_samples):
            if test_accept(0, 1):
                accepted +=1
        prob_accepted = accepted/num_samples
        
        self.assertGreater(prob_accepted, 0, "The accept function is not accepting any transitions to higher energy state")
        self.assertLess(prob_accepted, 1, "The accept function is accepting all transitions to higher energy state")

    def test_QeMCMC_gamma_zero(self):
        # Test the case where gamma = 0
        model_ = Model_Maker(6, "Fully Connected Ising", str(6) +" test case").model
        quant_MCMC = MCMC_qulacs(model_, 0, 5, 1, max_qubits = 6, CG_sample_number=1, naive = False)
        
        bad_count = 0
        for i in range(100):
            input_bitstring = ''.join(map(str, np.random.choice([0, 1], size=6)))
            s_prime = quant_MCMC.get_s_prime(input_bitstring)
            if s_prime != input_bitstring:
                bad_count +=1
        self.assertEqual(bad_count, 0, "The QeMCMC is not returning the same bitstring when gamma = 0")
    
    def test_CGQeMCMC_gamma_zero(self):
        # Test the case where gamma = 0
        model_ = Model_Maker(6, "Fully Connected Ising", str(6) +" test case").model
        quant_MCMC = MCMC_qulacs(model_, 0, 5, 1, max_qubits = 3, CG_sample_number=2, naive = False)
        
        bad_count = 0
        for i in range(100):
            input_bitstring = ''.join(map(str, np.random.choice([0, 1], size=6)))
            s_prime = quant_MCMC.get_s_prime(input_bitstring)
            if s_prime != input_bitstring:
                bad_count +=1
        self.assertEqual(bad_count, 0, "The CGQeMCMC is not returning the same bitstring when gamma = 0")
    
if __name__ == '__main__':
    unittest.main()


