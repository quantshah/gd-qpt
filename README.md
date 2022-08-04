# Gradient-descent quantum process tomography by learning Kraus operators

The code and data to reproduce the results of the following paper are available in this repository. Please cite the arXiv paper if you use this work and adapt the code further. Note that the code is still
being updated so expect some changes in the final version.

> Shahnawaz Ahmed, Fernando Quijandría, Anton Frisk Kockum, "Gradient-descent quantum process tomography by learning Kraus operators," (2022), [arXiv:2208.00812](https://arxiv.org/abs/2208.00812).

## Installation and use
The notebooks in the `examples` folder are standalone tutorials on how to reconstruct quantum processes with gradient-descent. These notebooks form the basis of the code that generates the data for various plots. The code depends on several open-soruce libraries that need to be installed to run the notebooks - Jax, dm-haiku, scipy, qutip.

I have prepared a python package `gd_qpt` that can be installed in development mode locally (such that any change in the code is instantly reflected). To install gd-qpt locally, clone/download the repository and run the following command from the gd-qpt folder:

```
pip install -e .
```

Then, run any of the notebooks in the example folder to run QPT on simulated data. 

Feel free to reach out to me at shahnawaz.ahmed95@gmail.com if you face any issues.

## Background

In GD-GPT, the quantum process reconstruction problem is framed as a learning task using measurement data from experiments, see below. The Kraus operator representation is chosen for completely positivitity (CP) of the process. The process is kept trace preserving (TP) during optimization with gradient descent using a retraction step that only performs updates on the so called Stiefel manifold keeping the process TP throughout the optimization [2].

The result is a flexible approach similar to the training of neural-networks with backpropagation that can match the performance of state-of-the-art process reconstrction methods such as projected least-squares (PLS) [3] and compressed sensing (CS) [4]. Similar to CS (but unlike PLS), GD-QPT can reconstruct a process from just a small number of random measurements, and similar to PLS (but unlike CS) it also works for larger system sizes, up to at least five qubits. We demonstrate results for both DV and CV systems. 

Bonus: Neural-network QPT - make a neural network generate the Kraus operators similar to previous works that use neural networks for quantum state tomography. However it seems there is no benefit to using neural networks and constrained gradient-descent with a good enough representation for the process works well.

<img width="402" alt="Screen Shot 2022-08-02 at 10 49 38 AM" src="https://user-images.githubusercontent.com/6968324/182333651-4a2060e1-30b6-48d3-ad84-8a3bea8d8b22.png">


## Continuous-variable QPT example

We use coherent states $|\alpha_i\rangle$ as probes and parity $\Pi(\beta_j)$ as the measurement operators with $\alpha_i, \beta_j$ being complex numbers representing points in the phase-space. Even with a very coarse probe and measurement grid, we get a reasonable reconstruction of the underlying process matrix.

<img width="422" alt="Screen Shot 2022-08-02 at 11 55 18 AM" src="https://user-images.githubusercontent.com/6968324/182347115-36209e9c-51ee-4622-9214-8601bdc19573.png">


## References

[1] Shahnawaz Ahmed, Fernando Quijandría, Anton Frisk Kockum, "Gradient-descent quantum process tomography by learning Kraus operators," (2022), [arXiv:2208.00812](https://arxiv.org/abs/2208.00812).

[2] H. D. Tagare, “Notes on optimization on Stiefel manifolds,” [Yale University, New Haven (2011)](https://noodle.med.yale.edu/hdtag/notes/steifel_notes.pdf)

[3] Adhikary, S., Srinivasan, S., Gordon, G. &amp; Boots, B.. (2020). Expressiveness and Learning of Hidden Quantum Markov Models. <i>Proceedings of the Twenty Third International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 108:4151-4161 Available from https://proceedings.mlr.press/v108/adhikary20a.html.

[4] Surawy-Stepney, Trystan, et al. "Projected least-squares quantum process tomography," (2021), [arXiv:2107.01060](https://arxiv.org/abs/2107.01060). See code implementation in [https://github.com/Hannoskaj/Hyperplane_Intersection_Projection](https://github.com/Hannoskaj/Hyperplane_Intersection_Projection).

[5] Rodionov, Andrey V., et al. "Compressed sensing quantum process tomography for superconducting quantum gates." [Physical Review B 90.14 (2014): 144504.](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.90.144504)

[6] Compressed sensing implementation from Qiskit Ignis: [https://github.com/Qiskit/qiskit-ignis/blob/101cdc01bee1be8cba3cc1103e5e146f43bfce96/qiskit/ignis/verification/tomography/fitters/cvx_fit.py](https://github.com/Qiskit/qiskit-ignis/blob/101cdc01bee1be8cba3cc1103e5e146f43bfce96/qiskit/ignis/verification/tomography/fitters/cvx_fit.py)

