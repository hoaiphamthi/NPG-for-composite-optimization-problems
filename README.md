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
  - `RERUN` (used when `LOAD_RESULTS = False`): `"all"` runs every algorithm (default). `"npg"` or `"competitors"` loads the saved results of an instance, re-runs only the NPG variants or only the other algorithms, and saves the merged results.

- The initial stepsize is determined via line search for the problems `lasso`, `min_length_curve`, `bcqp`, and `NMF`. For `maximum_likelihood`, `dual_max_entropy`, and `bcfp`, it is fixed at 0.001.

- We use `numpy.random.seed` to ensure reproducibility. Datasets labeled 1–10 correspond to random seeds 1–10. Problems with the same size and random seed should always produce identical results.

### Running the Experiments
- To obtain Figures 1–7, open the corresponding problem file (e.g., `lasso.py`) and set the control variables `LOAD_RESULTS = False` and `PLOT = False`. Then, in `run_experiment.py`, uncomment the relevant function call in `main()` (e.g., `experiment_lasso()`), and run the code.
- To obtain Figure 8, run the corresponding problem file (e.g., `lasso.py`, `bcqp.py`, etc.) with `LOAD_RESULTS = False`. Set `PLOT = True` to display the plots.

### Obtaining Plots from Presolved Results
- To reproduce Figures 1–8 using our presolved results, download the `results` folder from the following link:  
  https://drive.google.com/file/d/1wuAD7BE1xs8VM6OfscVwkUYEipfxPY9X/view  
  Place it inside this repository, then run the corresponding file as described above with `LOAD_RESULTS = True`.

### Experimental results
- The plots generated are saved in the plots folder
- The folder results contains the results of all the experiments.

### FISTA and the classical Armijo backtracking
- The script `fista_test.py` runs FISTA with backtracking (Beck and Teboulle, 2009; backtracking factor 1.5, 2 and 3) and the proximal gradient method with the classical Armijo backtracking (no increase of the stepsize, halving on failure) on nine instances of Lasso, Min-length and Dual-max-entropy (seed 1), with the same starting points, initial stepsize procedure and stopping criterion as in the paper.
- The results and a summary (`summary.txt`) are in the folders `results/FISTA_test` and `results/classic_armijo`. The comparison with NPG2 uses the main results of the corresponding instances (`results/<problem>/`, available from the link above).

### Revision note
- In `NPG` and `NPG_quad` (`alg.py`), the update of the stepsize two iterations back (`t_prev_prev`) has been corrected; all results and figures of the revised paper are computed with the corrected code.

### Experimental details
For Figures 8 and 9 in the paper, the detailed sizes and random seeds for each problem are provided here:
- Lasso                       : m = 1024, n = 8192, seed = 1   
- Min length curve problem    : m = 100, n = 10000, seed = 1 
- Maximum likelyhood problem  : n = 30, lb = 0.1, ub = 1000, M = 50, seed = 1
- Dual max entropy problem    : m = 4000, n = 5000, seed = 1
- BCQP                        : n = 1000, r = 10, seed = 1
- BCFP                        : n = 5000, r = 10, seed = 1
- NMF problem                 : m = 3000, n = 3000, r = 30, seed = 1
    

