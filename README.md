# TFE

KRR_notebook: new version of KRR + notebook form

CKN-MNIST: KRR on the dataset CKN-MNIST

New version: Modele implement the KRR, Optimiseur implement the different methods, Experiments does the experiments


New update: 
# Experiments

This directory contains all code needed to reproduce the figures presented in the report, except for the introductory quadratic example.

## Structure

- **`model/`**  
  Implements the models used in the experiments.

- **`optimizer/`**  
  Contains the optimization algorithms.

- **`algorithm/`**  
  Runs and evaluates the optimization algorithms on synthetic classification datasets (see figures from Chapter 4).

- **`figures/`**  
  Uses saved `.npy` files to generate the plots shown in Chapter 5.

- **`evolution/`**  
  Generates `.npy` files that track the training progress (e.g., loss, gradient norm) over time.

## Additional Folder

- **`quadratic/`** *(located outside this folder)*  
  Contains tests of optimization methods on a simple quadratic function and generates the figures used in the introduction.

