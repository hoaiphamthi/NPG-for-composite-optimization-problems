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
m = 1024
n = 8*m

def run_lasso(m = m, n = n, seed = seed):
    name = "lasso"
    np.random.seed(seed)
    tol = 1e-6
    N = 15000
    tau = 0.01
    name_instance = f"{name}_n={n}_m={m}_tau={str(tau).replace('.',',')}_seed={seed}"

    if LOAD_DATA:
        loaded_data = load(name_instance, name, saving_obj=1)
        A  = loaded_data["A"]
        b  = loaded_data["b"]
        x0 = loaded_data["x0"]
        xopt = loaded_data["xopt"]

    else:
        A = np.random.randn(m,n)
        xopt = np.random.randn(n) * np.random.binomial(1, 0.05, (n))
        b = A.dot(xopt) + np.random.normal(0,0.1,m)
        x0 = np.random.normal(size=n) 

    lamda = tau*np.max(A.T @ b)

    if SAVE_DATA:
        data = {"A":A, "b":b, "x0":x0, "xopt": xopt}
        save(data, name_instance, name,saving_obj=1)

    # ============================================================
    # Define objective, gradient and proximal operator
    # ============================================================
    def f(x):
        Ax_b = A.dot(x) - b
        return 0.5*(Ax_b.T.dot(Ax_b))
    def df(x):
        return A.T.dot(A.dot(x) - b)

    def oracle_f(x):
        return f(x), df(x)

    def g(x):
        return lamda*np.sum(np.abs(x))

    def prox_g(x, stepsize):
        return np.sign(x)*np.maximum(np.zeros(np.shape(x)),np.abs(x)-stepsize*lamda)


    def Run(algo, **kwargs):
        return algo(oracle_f, g, prox_g, x0, maxit = N, tol = tol, stop = "res", lns_init = True, verbose = False, 
                    track = ["res", "obj", "grad", "steps","time", "mse"], fixed_step = 0.001, xopt = xopt, **kwargs)

    def compare_signals(xopt, x):
        plt.figure(2)
        plt.clf()        
        plt.subplot(211)    
        plt.plot(xopt)
        plt.title('Original x',  fontsize=16)
        plt.subplot(212)
        plt.plot(x)
        plt.title('Reconstructed x',  fontsize=16)
        plt.tight_layout()
        plt.show()

    results = {}
    if LOAD_RESULTS:
            loaded_results = load(name_instance, name, saving_obj=2)
            for key, history in loaded_results.items():
                results[key] = history
    else : 
        runners = {
            "NPG1": lambda: Run(NPG, params = [0.1, 5.7, 0.7, 0.69], ver=1),
            "NPG2": lambda: Run(NPG, params = [0.1, 5.7, 0.99, 0.98], ver=2),
            "NPG_quad": lambda: Run(NPG_quad, params = [0.1, 5.7, 0.99, 0.98]),
            "AdPG": lambda: Run(AdProxGrad),
            "AdaPG" + str([3/2, 3/4]): lambda: Run(AdaPG, params=[3/2, 3/4]),
        }
        for param in [[1.1, 0.5], [1.2, 0.5]]:
            runners["PG-LS" + str(param)] = lambda param=param: Run(ProxGrad, params = param)
        results, xs = run_algorithms(runners, ['NPG1', 'NPG2', 'NPG_quad'], name_instance, name, RERUN)
        x2 = xs.get("NPG2")

    if SAVE_RESULTS and LOAD_RESULTS == False:
        save(results,name_instance,name, saving_obj=2)

    make_all_plots(results, name_instance, name, plot_mse=True, plot=PLOT)
    if PLOT and not LOAD_RESULTS: compare_signals(xopt, x2)
    parameters = {"additional_info":f"lambda = {lamda:.2f}" , "size":f"({m}, {n})", "seed": seed}

    return results, name_instance, name, parameters 

if __name__ == "__main__":
    run_lasso()



