Code used in the paper "Quantum-enhanced Markov Chain Monte Carlo for systems larger than your Quantum Computer"

It builds upon the numerics in Layden's work on the Quantum enhanced Markov Chain Monte Carlo (QeMCMC): https://www.nature.com/articles/s41586-023-06095-4, with the main contribution focussed on applying a "coarse graining" to the quantum proposal in order to analyse the possibility of some dampened quantum speedup remaining while the number of required qubits is lowered. 

## Code details
The code is essentially built upon another repository: https://github.com/pafloxy/quMCMC, with changes made to allow for different Coarse graining of the Ising model.


An example experiment is given in the folder titled "MCMC", where a Markov Chain Monte Carlo algorithm is run for an example 9 spin instance. Classical "Uniform" and "local" update proposals are compared with a CGQeMCMC implimentationusing only 3 simulated qubits. Even though the number of qubits is quadratically reduced from the number requred by the original algorithm of Layden et al, we still see quantum speedup. The follwoing results can be viewed by running Analyse_all.py. See [Ferguson 2024] for details.


![9_spin_mag_example](https://github.com/Stuartferguson00/CGQeMCMC/assets/99267485/0dba491f-92c6-40b6-833c-7e5af80286d9)

![9_spin_E_example](https://github.com/Stuartferguson00/CGQeMCMC/assets/99267485/3623e7ca-b74e-4448-95dd-6d211dcee5b7)


### To recreate the results:
- Ensure the necessary python libraries are installed.
- Delete the result files in results/1_0/
- From terminal, run the python file: "Run_class.py" which will initalise classical Markov chains. Command line argument required are:
  - n_spins: (int) Number of spins in system. (9 in given example) 
  - temp: (float) Temperature of system. (1 in given example) 
  - reps: (int) Number of markov chains to condider. (10 in given example)
  - n_hops: (int) Length of chains to consider. (10000 in given example)
- From terminal, run the python file: "Run_quant.py" which will initalise Quantum Markov chains Markov chain. Command line arguments required are:
  - n_spins: (int) As above
  - temp: (float)  As above
  - reps: (int) As above
  - n_hops: (int) As above
  - mult_samp: (bool) Whether to use multiple groups in Coarse graining (True in given example)
  - group_size: (int) Size of group to use in coarse graining (3 in given example)
- Run Analyse_all.py to view results.

Please bear in mind this is a code built for experimentation and does not have sophistocated error-catching capabilities. Likewise the example code analysis code is written for the specific inputs given above.
Also note that the above code currently uses simple cpu parallelisation to run multiple Markov chains concurrently, so running running code with multiple reps will impact the performance of your computer.

The quantum simulator used is Qulacs.


## References
Quantum-enhanced Markov Chain Monte Carlo for systems larger than your Quantum Computer by S. Ferguson and P. Wallden
Quantum-enhanced Markov chain Monte Carlo by David Layden et al.
Qulacs Simulator

## License
The package is licensed under  MIT License
