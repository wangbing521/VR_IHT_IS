"""
Run the finite support identification experiment (SARAH-IHT).

Generates synthetic sparse regression data with three signal regimes
(small / medium / large), runs SARAH-IHT with importance sampling, and saves all results to JSON files.

Key design: starts from a deliberately WRONG initial support (no overlap
with the true support), forcing the algorithm to discover the correct
support over multiple iterations.

Usage:
    python run_support_experiment.py
"""

import numpy as np
import os
import json
import time

from importance_sampling import Importance_Sampling


D = 500
N = 5000
S_TRUE = 50
LAM = 1e-3
SIGMA = 0.05
MAX_ITER = 2000
ETA = 4
TOL = 1e-4
N_RUNS = 20
WRONG_AMP = 0.1

ALPHA_MAP = {
    "small":  0.1,
    "medium": 0.4,
    "large":  1.0,
}

METHODS = ["IS"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    if isinstance(obj, list):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, tuple):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj


def save_json(data, filename):
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, ensure_ascii=False, indent=2)
    print(f"  Saved: {filepath}")


def run_single_experiment(A, b, S_nat, sparsity, seed, L, x0, pr=False):
    model = Importance_Sampling(eta=ETA, lam=LAM, maxIter=MAX_ITER, seed=seed)
    model.fit(A=A, b=b, S_nat=S_nat, sparsity=sparsity, L=L, x0=x0.copy(), pr=pr)

    return {
        "method": "IS",
        "obj": model.obj,
        "time": model.Time,
        "iter": model.iter_log,
        "N_switch": model.N_switch,
        "M_nat": model.M_nat,
        "gamma_hat": model.gamma_hat,
        "T_stab": model.T_stab,
    }


def aggregate_runs(all_run_results):
    """
    Given a list of run result dicts, compute mean ± std for array fields,
    and keep the list of scalar fields (like T_stab).
    """
    n_runs = len(all_run_results)
    if n_runs == 0:
        return {}

    keys = all_run_results[0].keys()
    aggregated = {}

    for key in keys:
        values = [r[key] for r in all_run_results]

        if key in ("method",):
            aggregated[key] = values[0]
            continue

        if key == "T_stab":
            aggregated[key] = values
            continue

        if isinstance(values[0], list):
            max_len = max(len(v) for v in values)
            arr = np.full((n_runs, max_len), np.nan)
            for i, v in enumerate(values):
                arr[i, :len(v)] = np.asarray(v, dtype=float)
            aggregated[key] = np.nanmedian(arr, axis=0)
            aggregated[key + "_q1"] = np.nanpercentile(arr, 25, axis=0)
            aggregated[key + "_q3"] = np.nanpercentile(arr, 75, axis=0)
        else:
            aggregated[key] = values

    return aggregated


def main():
    print("=" * 70)
    print("Finite Support Identification Experiment  (SARAH-IHT)")
    print(f"  d={D}, N={N}, s={S_TRUE}, lambda={LAM}, sigma={SIGMA}")
    print(f"  maxIter={MAX_ITER}, eta={ETA}, n_runs={N_RUNS}")
    print(f"  wrong_amp={WRONG_AMP}")
    print("=" * 70)

    data_seed = 42
    rng_data = np.random.default_rng(data_seed)

    A = rng_data.standard_normal((N, D))
    S_nat = tuple(sorted(rng_data.choice(D, size=S_TRUE, replace=False)))
    signs = rng_data.choice([-1, 1], size=S_TRUE)
    noise = SIGMA * rng_data.standard_normal(N)

    wrong_pool = list(set(range(D)) - set(S_nat))

    L_lip = np.linalg.norm(A.T @ A / N + LAM * np.eye(D), ord=2)
    print(f"  Lipschitz constant L = {L_lip:.4f}")
    print(f"  Effective step = 1/({ETA}*L) = {1.0/(ETA*L_lip):.4f}")

    all_results = {}

    for regime_name, alpha in ALPHA_MAP.items():
        print(f"\n{'─' * 70}")
        print(f"Regime: {regime_name}  (alpha = {alpha})")
        print(f"{'─' * 70}")

        x_nat = np.zeros(D)
        x_nat[list(S_nat)] = alpha * signs
        b = A @ x_nat + noise

        for method in METHODS:
            label = "VR-IHT-IS" if method == "IS" else "VR-IHT-US"
            print(f"\n  Method: {label}")
            run_results = []

            for run_id in range(N_RUNS):
                init_seed = 1000 + run_id
                rng_init = np.random.default_rng(init_seed)
                wrong_supp = rng_init.choice(wrong_pool, size=S_TRUE, replace=False)
                x0 = np.zeros(D)
                x0[wrong_supp] = WRONG_AMP * rng_init.choice([-1, 1], size=S_TRUE)

                seed = (list(ALPHA_MAP.keys()).index(regime_name) * 1000
                        + METHODS.index(method) * 100 + run_id)

                t0 = time.time()
                result = run_single_experiment(
                    A, b, S_nat, S_TRUE, seed, L_lip, x0,
                    pr=(run_id == 0)
                )
                elapsed = time.time() - t0
                print(f"    Run {run_id+1:2d}/{N_RUNS}  "
                      f"T_stab={result['T_stab']:4d}  "
                      f"N_switch={result['N_switch'][-1]:4d}  "
                      f"M_nat_final={result['M_nat'][-1]:3d}  "
                      f"time={elapsed:.1f}s")
                run_results.append(result)

            key = f"{regime_name}_{method}"
            agg = aggregate_runs(run_results)
            agg["alpha"] = alpha
            agg["regime"] = regime_name
            agg["method_name"] = method
            agg["S_nat"] = list(S_nat)
            agg["config"] = {
                "d": D, "N": N, "s_true": S_TRUE,
                "lam": LAM, "sigma": SIGMA,
                "maxIter": MAX_ITER, "eta": ETA,
                "wrong_amp": WRONG_AMP,
                "n_runs": N_RUNS
            }
            all_results[key] = agg

    print(f"\n{'═' * 70}")
    print("Saving results...")
    save_json(all_results, "support_experiment_all.json")

    print(f"\n{'═' * 70}")
    print("Experiment complete!")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
