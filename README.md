# BayesXII

## Introduction
BayesXII is a fast parallelized algorithm for Bayesian regression models, where "X" stands for Bayesian alphabet methods and "II" stands for "parallel". Ideally, the computations at each step of the MCMC chain can be accelerated by *k* times, where *k* is the number of computer processors, up to *p* times, where *p* is the number of markers. Ideally, the sampling of each marker effect can be made independent of samples for other marker effects within each step of the chain. 

Details can be found in this [paper](https://gsejournal.biomedcentral.com/articles/10.1186/s12711-020-00533-x):  
*Zhao T, Fernando R, Garrick D, et al. Fast parallelized sampling of Bayesian regression models for whole-genome prediction[J]. Genetics Selection Evolution, 2020, 52(1): 1-11.*

All codes are provided here, and a demo using a small dataset is provided. The parallel implementation of the BayesXII algorithm was based on Message Passing Interface (MPI), and the tutotial of using MPI in Julia is provided in my repository named [MPI_testing](https://github.com/zhaotianjing/MPI_testing/blob/master/MPI.ipynb).

## Play with the demo

Step 1. run *create_data_demo.jl* to create data such as genotype and phenotype. Output data are in total about 3 MB, which are also provided in the *demo* folder.  
Step 2. run *split_column.jl* to create data specifically needed by different processors. Output data are in total 2 MB, which are also provided in the *demo* folder.  
Step 3. run the MPI parallel code *BayesXII.jl* on multiple processors. A slurm submit file is provided in the *demo* folder to submit parallel jobs.

## Contact
Feel free to contact Hao Cheng [qtlcheng@ucdavis.edu] and Tianjing Zhao [tjzhao@ucdavis.edu] for any problem.