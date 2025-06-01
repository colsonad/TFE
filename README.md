# Thesis Experiments Code Repository

This repository contains the source code, configuration files, and scripts used to conduct the experiments described in the Master's thesis:

    "Generalization Performance with Non-Constant Stepsizes"
    Author: Adeline Colson
    Institution: École polytechnique de Louvain
    Academic Year: 2024–2025

This codebase supports the experimental results presented in the thesis and reproduces figures and evaluations referenced throughout the thesis.

# Repository structure
## Folder Experiments/

- **`Modele.py`**  
  Implements the KRR model used in the experiments.

- **`Optimiseur.py`**  
  Contains all the optimization algorithms presented in the thesis.

- **`Algo.ipynb`**  
  Runs and evaluates the optimization algorithms on synthetic classification datasets (see figures from Chapter 4).

- **`Figures.ipynb`**  
  Uses saved `.npy` files to generate the plots shown in Chapter 5.

- **`Evolution.ipynb`**  
  Generates `.npy` files that track the training progress (e.g., training function, gradient norm, test accuracy) over time.

## Additional File

- **`Quadratic.ipynb`**   
  Contains tests of optimization methods on a simple quadratic function and generates the figures used in the introduction.

- **`Kernel_tuning.ipynb`**  
  Contains the tuning of the RBF parameter for the CKN-MNIST dataset.

- **`Algorithms.ipynb`**  
  Contains experiments on constant gradient descent on the CKN-MNIST dataset.

- **`New_dataset.ipynb`**  
  Contains same expirements as in kernel_tuning and Algorithms, but for the a4a dataset.

- **`Modele.ipynb`**  
  Implements the models used in the experiments.

- **`Optimiseur.ipynb`**  
  Contains the optimization algorithms presented in the thesis.