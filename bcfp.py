import numpy as np
from save_and_plot import save, load, make_all_plots, run_algorithms
from alg import AdProxGrad, ProxGrad, NPG, NPG_quad, AdaPG
import matplotlib.pyplot as plt

# Control section
# - If data is not loaded then it will be created.
# - If results is not loaded then the algorithms will run.
######################################################
LOAD_DATA = False
SAVE_DATA = False
LOAD_RESULTS = True
SAVE_RESULTS = True
PLOT = False
RERUN = "all"   # "all": run every algorithm; "npg" / "competitors": re-run only the NPG variants / only the other algorithms and merge them into the saved results
######################################################

seed = 1
n = 5000
r = 10
N = 2000
def run_bcfp( n = n, r = r, seed = seed):
    name = "bcfp"
    np.random.seed(seed)
    tol = 1e-6

    name_instance = f"{name}_n={n}_r={r}_seed={seed}"

    if LOAD_DATA:
        loaded_data = load(name_instance, name, saving_obj=1)
        A_eq  = loaded_data["A"]
        b_eq  = loaded_data["b"]
        x0 = loaded_data["x0"]
    else:
        U = np.random.randn(n, n)
        A_mat = U.T @ np.diag(np.random.uniform(-1, r, n)) @ U 
        b_vec = np.random.uniform(low=1, high=10, size=n)
        c0 = np.random.uniform(low=1, high=10)
        e_vec = np.random.uniform(low=1, high=10, size=n)
        d0 = np.random.uniform(low=1, high=10)
        x0 = np.random.uniform(low=1, high=10, size=n)

    # ============================================================
    # Define objective, gradient and proximal operator
    # ============================================================
    def f(x):
        num = x.T @ (A_mat @ x) + b_vec.T @ x + c0
        den = e_vec.T @ x + d0
        return num / den

    def df(x):
        num = x.T @ (A_mat @ x) + b_vec.T @ x + c0
        den = e_vec.T @ x + d0
        grad_num = 2.0 * (A_mat @ x) + b_vec
        grad_frac = (grad_num * den - num * e_vec) / (den ** 2)
        return grad_frac

    def oracle_f(x):
        return f(x), df(x)

    def g(x):
        return 0.0

    def prox_g(x, stepsize):
        return np.clip(x, 0, 1)

    def Run(algo, **kwargs):
        return algo(oracle_f, g, prox_g, x0, maxit = N, tol = tol, stop = "res", lns_init = False, verbose = False, 
                    track = ["res", "obj", "grad", "steps","time"], fixed_step = 0.001, **kwargs)

    results = {}
    if LOAD_RESULTS:
        loaded_results = load(name_instance, name, saving_obj=2)
        for key, history in loaded_results.items():
            results[key] = history
    else:
        runners = {
            "NPG2": lambda: Run(NPG, params = [0.1, 5.7, 0.99, 0.98], ver=2),
            "AdPG": lambda: Run(AdProxGrad),
            "AdaPG" + str([3/2, 3/4]): lambda: Run(AdaPG, params=[3/2, 3/4]),
        }
        for param in [[1.1, 0.5], [1.2, 0.5]]:
            runners["PG-LS" + str(param)] = lambda param=param: Run(ProxGrad, params = param)
        results, xs = run_algorithms(runners, ['NPG2'], name_instance, name, RERUN)

    if SAVE_RESULTS and LOAD_RESULTS == False:
        save(results,name_instance,name, saving_obj=2)

    parameters = {"additional_info": '', "size":f"({n}, {r})", "seed": seed}
    make_all_plots(results, name_instance, name, plot=PLOT)
    return results, name_instance, name, parameters

if __name__ == "__main__":
    run_bcfp()
