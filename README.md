This document provides instructions for the code accompanying the paper  
**"Composite optimization problems via novel  proximal gradient
algorithms  and applications"**  
by Pham Thi Hoai and Nguyen Pham Duy Thai.

### Introduction
- There are seven Python files, each corresponding to a tested problem instance.
- The file `run_experiment.py` is designed to run multiple problems, sizes, and random datasets in a single execution.
- In the `main()` function, call the function corresponding to the problem you want to run (e.g., `experiment_lasso()`). The problem sizes and random seeds should be specified within each experiment function.

- Each `.py` file begins with a control section that must be properly configured:
  - `LOAD_DATA`: If `True`, the code loads an existing dataset; otherwise, a new dataset is generated.
  - `SAVE_DATA`: If `True`, the generated data is saved. Note that datasets can be large and may consume significant memory.
  - `LOAD_RESULTS`: If `True`, previously saved results are loaded; otherwise, the algorithms are executed and results are generated.
  - `SAVE_RESULTS`: If `True`, results from executed problems are saved. It is highly recommended to set this to `True`.

- The initial stepsize is determined via line search for the problems `lasso`, `min_length_curve`, `bcqp`, and `NMF`. For `maximum_likelihood`, `dual_max_entropy`, and `bcfp`, it is fixed at 0.001.

- We use `numpy.random.seed` to ensure reproducibility. Datasets labeled 1–10 correspond to random seeds 1–10. Problems with the same size and random seed should always produce identical results.

### Running the Experiments
- To obtain Figures 1–7, open the corresponding problem file (e.g., `lasso.py`) and set the control variables `LOAD_RESULTS = False` and `PLOT = False`. Then, in `run_experiment.py`, uncomment the relevant function call in `main()` (e.g., `experiment_lasso()`), and run the code.
- To obtain Figure 8, run the corresponding problem file (e.g., `lasso.py`, `bcqp.py`, etc.) with `LOAD_RESULTS = False`. Set `PLOT = True` to display the plots.

### Obtaining Plots from Presolved Results
- To reproduce Figures 1–8 using our presolved results, download the `results` folder from the following link:  
  https://drive.google.com/file/d/1l4lDUnLHnA31VLDydmYeSmzNQyEt9TjV/view  
  Place it inside this repository, then run the corresponding file as described above with `LOAD_RESULTS = True`.

### Experimental results
- The plots generated are saved in the plots folder
- The folder results contains the results of all the experiments.

### Experimental details
For Figure 8 in the paper, the detailed sizes and random seeds for each problem are provided here:
- Lasso                       : m = 1024, n = 8192, seed = 1   
- Min length curve problem    : m = 100, n = 10000, seed = 1 
- Maximum likelyhood problem  : n = 30, lb = 0.1, ub = 1000, M = 50, seed = 1
- Dual max entropy problem    : m = 4000, n = 5000, seed = 1
- BCQP                        : n = 1000, r = 10, seed = 1
- BCFP                        : n = 5000, r = 10, seed = 1
- NMF problem                 : m = 3000, n = 3000, r = 30, seed = 1
    
