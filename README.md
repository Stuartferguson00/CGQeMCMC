Code used in the paper "Quantum-enhanced Markov Chain Monte Carlo for systems larger than your Quantum Computer": https://arxiv.org/abs/2405.04247

It builds upon the numerics in Layden's work on the Quantum enhanced Markov Chain Monte Carlo (QeMCMC): https://www.nature.com/articles/s41586-023-06095-4, with the main contribution focussed on applying a "coarse graining" to the quantum proposal in order to analyse the possibility of some dampened quantum speedup remaining while the number of required qubits is lowered. 

## Code details

The code is essentially built upon another repository: https://github.com/pafloxy/quMCMC, with changes made to allow for different Coarse graining of the Ising model.

An example experiment is given in the folder titled "MCMC", where a Markov Chain Monte Carlo algorithm is run for an example 9 spin instance. Classical "Uniform" and "local" update proposals are compared with a CGQeMCMC implimentationusing only 3 simulated qubits. Even though the number of qubits is quadratically reduced from the number requred by the original algorithm of Layden et al, we still see quantum speedup. The follwoing results can be viewed by running Analyse_all.py. See [Ferguson 2024] for details.



![9_spin_E_example](9_spin_T_1.png)


### To recreate the results:

Please bear in mind this is a code built for experimentation and does not have sophistocated error-catching capabilities. Likewise the example code analysis code is written for specific inputs.

Also note that the above code currently uses simple cpu parallelisation to run multiple Markov chains concurrently, so running running code with multiple reps will impact the performance of your computer.

The quantum simulator used is Qulacs.


## References
Quantum-enhanced Markov Chain Monte Carlo for systems larger than your Quantum Computer by S. Ferguson and P. Wallden https://arxiv.org/abs/2405.04247

Quantum-enhanced Markov chain Monte Carlo by David Layden et al. https://www.nature.com/articles/s41586-023-06095-4

Qulacs Simulator https://quantum-journal.org/papers/q-2021-10-06-559/

## License
The package is licensed under  MIT License
