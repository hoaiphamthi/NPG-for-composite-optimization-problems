# FISTA with backtracking (Beck and Teboulle, 2009) and the proximal gradient method with the classical Armijo
# backtracking (stepsize never increased, halved on failure: ProxGrad with params [1, 0.5]) on nine convex instances.
#
# The instances are built exactly as in the problem files (same data, starting point, tolerance and N_max).
# Stopping criterion as in the paper: residual (gradient mapping at the returned point) <= tol. A run that has not
# converged within the paper's N_max is continued while it still makes progress, up to 20 N_max iterations or 180 s,
# to obtain the actual number of iterations. On Dual-max-entropy, FISTA and PG-LS(1, 0.5) start with the initial line
# search (their stepsizes cannot increase; the paper's methods start there from the fixed stepsize 0.001).
#
# Output: results/FISTA_test and results/classic_armijo (one .npy file per instance and summary.txt).
# NPG2 is read from the main results results/<problem>/ (available from the link in README.md).
import os
import time
import numpy as np
import alg
import lasso, min_length_curve, dual_max_entropy
from alg import FISTA, ProxGrad

ETAS = [1.5, 2.0, 3.0]
EXTEND, MAX_SECONDS = 20, 180
INSTANCES = [
    (lasso, "run_lasso", (512, 1024, 1)), (lasso, "run_lasso", (512, 2048, 1)), (lasso, "run_lasso", (1024, 2048, 1)),
    (min_length_curve, "run_min_len_curve", (500, 5000, 1, 1500)), (min_length_curve, "run_min_len_curve", (2000, 5000, 1, 200)),
    (min_length_curve, "run_min_len_curve", (1000, 10000, 1, 1500)),
    (dual_max_entropy, "run_dual_max_entropy", (500, 2000, 1, 100)), (dual_max_entropy, "run_dual_max_entropy", (2000, 4000, 1, 200)),
    (dual_max_entropy, "run_dual_max_entropy", (4000, 5000, 1, 200)),
]
FOLDER = {lasso: "lasso", min_length_curve: "min_len_curve", dual_max_entropy: "dual_max_ent"}


class Captured(Exception):
    pass


class Stop(Exception):
    pass


def closure_vars(fun):
    return dict(zip(fun.__code__.co_freevars, (c.cell_contents for c in fun.__closure__)))


def build_instance(module, fn, args):
    """Build the instance exactly as the problem file does; return its name and data, without running or saving anything."""
    got = {}
    def grab(runners, npg_keys, name_instance, name, rerun):
        got.update(runners=runners, name_instance=name_instance)
        raise Captured
    module.run_algorithms, module.SAVE_RESULTS, module.LOAD_RESULTS, module.PLOT = grab, False, False, False
    module.SAVE_DATA = False
    try:
        getattr(module, fn)(*args)
    except Captured:
        pass
    data = closure_vars(closure_vars(got["runners"]["AdPG"])["Run"])   # oracle_f, g, prox_g, x0/y0, N, tol, ...
    return got["name_instance"], data


def run(module, data, algo, params):
    """Run until res <= tol; past N_max, stop if stalled (best residual of the last window not 5% below the best
    before it, window = max(100, k/10)), at EXTEND * N_max iterations, or after MAX_SECONDS."""
    N = data["N"]
    state = {"t0": time.time()}
    original = alg.collect_history
    def watched(history, info):
        history = original(history, info)
        k, res = len(history["res"]), history["res"]
        if info["res"] > data["tol"] and k > N:
            w = max(100, k // 10)
            if k % 100 == 0 and k > N + w and min(res[-w:]) > 0.95 * min(res[:-w]):
                raise Stop(history)
            if time.time() - state["t0"] > MAX_SECONDS:
                raise Stop(history)
        return history
    alg.collect_history = watched
    x0 = data["x0"] if "x0" in data else data["y0"]
    kw = dict(maxit=EXTEND * N, tol=data["tol"], stop="res", lns_init=True, verbose=False, fixed_step=0.001, params=params)
    if algo is FISTA:
        kw["res_at"] = "x"
    if module is lasso:
        kw.update(track=["res", "obj", "grad", "steps", "time", "mse"], xopt=data["xopt"])
    try:
        h = algo(data["oracle_f"], data["g"], data["prox_g"], x0, **kw)[1]
    except Stop as e:
        h = e.args[0]
    finally:
        alg.collect_history = original
    return h



def it_to(res, tol):
    return next((i + 1 for i, r in enumerate(res) if 0 < r <= tol), None)


def write_summaries(out_f, out_a):
    """summary.txt of both folders, from the saved runs."""
    hdr = ("Same instances, starting points, initial stepsize procedure and tolerance as in the paper. Residual: for FISTA the\n"
           "gradient mapping at its returned point x_k; for NPG2 and PG-LS the residual ||x_{k+1} - x_k|| / t_k of the paper.\n"
           "NPG2: from the main result files results/<problem>/ (available from the link in README.md).\n")
    fl = ["FISTA with backtracking (Beck-Teboulle 2009), backtracking factor eta in {1.5, 2, 3}; stopping criterion residual <= 1e-6.\n"
          "Runs not converged within the paper's N_max were continued (up to 20 N_max or 180 s) to obtain the actual iteration count.\n"
          "On Dual-max-entropy FISTA uses the initial line search (its stepsize cannot increase; the paper's methods start from 0.001).\n" + hdr]
    al = ["Proximal gradient method with the classical Armijo backtracking: each search starts from the last accepted stepsize\n"
          "(no increase) and halves it until the descent condition holds, PG-LS(1, 0.5).\n"
          "Reported: iterations to reach residual <= 1e-5, counting only the iterations before the stepsize falls below 1e-8. Near the\n"
          "solution the descent test can fail because of rounding errors, and since the stepsize cannot increase it then collapses;\n"
          "the residuals after that point are not meaningful. On Dual-max-entropy the stepsize is reduced to ~1e-4 in the first\n"
          "iterations (large curvature at the starting point) and cannot increase afterwards; the residual stays above 1.\n" + hdr]
    for module, fn, args in INSTANCES:
        name, _ = build_instance(module, fn, args)
        npg = np.load(os.path.join("results", FOLDER[module], name + ".npy"), allow_pickle=True).item()["NPG2"]
        fista = np.load(os.path.join(out_f, name + ".npy"), allow_pickle=True).item()
        armijo = np.load(os.path.join(out_a, name + ".npy"), allow_pickle=True).item()["PG-LS(1, 0.5)"]
        n2 = len(npg["res"])
        fl.append(f"\n{name}\n   NPG2            {n2:6d} it {npg['time'][-1]:8.3f} s")
        for k, h in fista.items():
            n = it_to(h["res"], 1e-6)
            fl.append(f"   {k:15s} " + (f"{n:6d} it {h['time'][n-1]:8.3f} s   ({n/n2:.1f} x NPG2 iterations)" if n else
                      f"not reached: {len(h['res'])} it, {h['time'][-1]:.0f} s, final residual {h['res'][-1]:.1e}"))
        steps = armijo["steps"]
        col = next((i for i, s in enumerate(steps) if s < 1e-8), len(steps))
        pre = [r for r in armijo["res"][:col] if r > 0]
        a5, b5 = it_to(armijo["res"][:col], 1e-5), it_to(npg["res"], 1e-5)
        al.append(f"\n{name}\n   NPG2            residual <= 1e-5 after {b5:5d} it")
        if a5:
            al.append(f"   PG-LS(1, 0.5)   residual <= 1e-5 after {a5:5d} it   ({a5/b5:.1f} x NPG2)")
        elif col < len(steps):
            al.append(f"   PG-LS(1, 0.5)   residual <= 1e-5 not reached: stepsize collapse at iteration {col + 1}, best residual before it {min(pre):.1e}")
        else:
            al.append(f"   PG-LS(1, 0.5)   residual <= 1e-5 not reached ({len(armijo['res'])} it, {armijo['time'][-1]:.0f} s); best residual {min(pre):.1e}")
    open(os.path.join(out_f, "summary.txt"), "w").write("\n".join(fl) + "\n")
    open(os.path.join(out_a, "summary.txt"), "w").write("\n".join(al) + "\n")


if __name__ == "__main__":
    import sys
    out_f, out_a = os.path.join("results", "FISTA_test"), os.path.join("results", "classic_armijo")
    if "--summary-only" not in sys.argv:          # run all tests (about 20 minutes), then write the summaries
        os.makedirs(out_f, exist_ok=True); os.makedirs(out_a, exist_ok=True)
        for module, fn, args in INSTANCES:
            name, data = build_instance(module, fn, args)
            np.save(os.path.join(out_f, name + ".npy"), {f"FISTA(eta={eta:g})": run(module, data, FISTA, [eta]) for eta in ETAS})
            np.save(os.path.join(out_a, name + ".npy"), {"PG-LS(1, 0.5)": run(module, data, ProxGrad, [1.0, 0.5])})
            print(name, "done", flush=True)
    write_summaries(out_f, out_a)
