"""Run the heterogeneous sparse linear regression experiment."""

import json
import os
import time

import numpy as np

from heterogeneous import (
    compute_global_lipschitz,
    describe_heterogeneity,
    generate_synthetic_data,
)
from uniform_sampling import Uniform_Sampling
from importance_sampling import Importance_Sampling


CONFIG = {
    "N": 6000,
    "d": 200,
    "s": 20,
    "sigma_list": [0.3, 0.5, 0.7, 1.0],
    "n_trials": 20,
    "max_iter": 300,
    "batch_size": 20,
    "m": None,
    "eta": 4.0,
    "tol": 1e-4,
    "seed_base": 42,
}


def run_single_experiment(sigma, trial, config):
    N = config["N"]
    d = config["d"]
    s = config["s"]
    batch_size = config["batch_size"] or int(np.sqrt(N))
    m = config["m"] or max(1, 2 * batch_size)
    data_seed = config["seed_base"] * 10_000 + int(sigma * 100) * 100 + trial
    algorithm_seed = data_seed + 1_000_000

    alphas, centers, targets, x_true = generate_synthetic_data(
        N=N, d=d, s=s, sigma=sigma, seed=data_seed
    )
    heterogeneity = describe_heterogeneity(alphas)
    L_avg = compute_global_lipschitz(centers)
    L_max = float(np.max(alphas))

    vr_iht = Uniform_Sampling(
        eta=config["eta"], max_iter=config["max_iter"], tol=config["tol"]
    ).fit(
        centers, targets, x_true, s, batch_size=batch_size, m=m,
        seed=algorithm_seed + 1, verbose=trial == 0,
    )
    vr_iht_is = Importance_Sampling(
        eta=config["eta"], max_iter=config["max_iter"], tol=config["tol"]
    ).fit(
        centers, alphas, targets, x_true, s, batch_size=batch_size, m=m,
        seed=algorithm_seed + 2, verbose=trial == 0,
    )

    res_vr = vr_iht.get_results()
    res_is = vr_iht_is.get_results()
    print(
        f"sigma={sigma:.1f}, trial={trial:02d}, "
        f"Lmax/Lmin={heterogeneity['ratio']:.1f}, "
        f"F_unif={res_vr['obj'][-1]:.3e}, "
        f"F_IS={res_is['obj'][-1]:.3e}"
    )
    return {
        "sigma": sigma,
        "trial": trial,
        "N": N,
        "d": d,
        "s": s,
        "batch_size": batch_size,
        "m": m,
        "max_iter": config["max_iter"],
        "heterogeneity": heterogeneity,
        "L_avg": L_avg,
        "L_max": L_max,
        "VR_IHT": {
            "obj": res_vr["obj"].tolist(),
            "ngrad": res_vr["ngrad"].tolist(),
            "time": res_vr["time"].tolist(),
            "dist_to_opt": res_vr["dist_to_opt"].tolist(),
            "f_opt": res_vr["f_opt"],
        },
        "VR_IHT_IS": {
            "obj": res_is["obj"].tolist(),
            "ngrad": res_is["ngrad"].tolist(),
            "time": res_is["time"].tolist(),
            "dist_to_opt": res_is["dist_to_opt"].tolist(),
            "f_opt": res_is["f_opt"],
        },
    }


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    records = []
    start_time = time.time()

    for sigma in CONFIG["sigma_list"]:
        for trial in range(CONFIG["n_trials"]):
            records.append(run_single_experiment(sigma, trial, CONFIG))

    output_path = os.path.join(results_dir, "heterogeneous_experiments.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(records, output_file, indent=2)
    elapsed = time.time() - start_time
    print(f"Saved {len(records)} records to {output_path} in {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()