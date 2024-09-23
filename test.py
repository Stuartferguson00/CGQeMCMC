import unittest
import cgqemcmc

class TestMCMC(unittest.TestCase):
    
    
    
    def test_proposal_distribution(self):
        # Example test case for the proposal distribution
        samples = [proposal_distribution() for _ in range(1000)]
        mean = sum(samples) / len(samples)
        self.assertAlmostEqual(mean, 0, delta=0.1)  # Assuming a mean of 0
    
    def test_mcmc_output(self):
        # Example test case for the MCMC output
        np.random.seed(42)  # Fixing the seed for reproducibility
        chain = run_mcmc(steps=1000)
        self.assertEqual(len(chain), 1000)

if __name__ == '__main__':
    unittest.main()
