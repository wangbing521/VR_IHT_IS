import sys
import os
import json
import random
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from matplotlib.dates import AutoDateFormatter, AutoDateLocator

from Trust_Region import Trust_Region
from importance_sampling import Importance_Sampling
from data import load_groups_csi300, load_csi300, build_group_idx_csi300


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.4,
    "figure.figsize": (6.4, 4.8),
    "figure.dpi": 180,
    "savefig.dpi": 350,
    "savefig.bbox": "tight",
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.8,
    "grid.linestyle": "-",
})


def average_obj_list(obj_runs):
    """
    obj_runs: list of list
    Pad objective trajectories with NaN before averaging when their lengths differ.
    """
    max_len = max(len(x) for x in obj_runs)
    arr = np.full((len(obj_runs), max_len), np.nan, dtype=float)

    for i, obj in enumerate(obj_runs):
        arr[i, :len(obj)] = np.asarray(obj, dtype=float)

    return np.nanmean(arr, axis=0).tolist()


def average_w_list(w_runs):
    """
    w_runs: list of ndarray/list
    Compute the element-wise mean of the model weights.
    """
    arr = np.asarray(w_runs, dtype=float)
    return np.mean(arr, axis=0)


dataset = load_csi300(download=False)
groups = load_groups_csi300(download=False)
lookup = build_group_idx_csi300(dataset, groups)

npdata = dataset.values

scaler_X = StandardScaler()

split = int(0.8 * npdata.shape[0])
data_train, data_test = npdata[:split, :], npdata[split:, :]

X_train, y_train = data_train[:, :-1], data_train[:, -1]
X_test, y_test = data_test[:, :-1], data_test[:, -1]

y_offset = np.mean(y_train)

X = scaler_X.fit_transform(X_train)
y = y_train - y_offset

print("X shape:", X.shape)


lam = 1

maxIter = 1800
tol = 1e-4

n = X.shape[0]
d = X.shape[1]
w0 = np.zeros(d)
sparsity = int(0.3 * d)

batch_size = int(np.sqrt(n))
m = 2 * batch_size

L_X = (2.0 / n) * (np.linalg.norm(X, ord=2) ** 2)
eta = 4.0

n_runs = 20


is_obj_runs = []
is_w_runs = []
is_time_runs = []

tr_obj_runs = []
tr_w_runs = []
tr_time_runs = []

for run_id in range(n_runs):
    print(f"\n========== Run {run_id + 1}/{n_runs} ==========")

    seed_base = 42
    random.seed(seed_base + run_id)
    np.random.seed(seed_base + run_id)

    IS = Importance_Sampling(w0=w0.copy(), eta=eta, lam=lam, maxIter=maxIter, tol=tol)
    t_start = time.time()
    IS.fit(
        X_train=X,
        y_train=y,
        batch_size=batch_size,
        L=L_X + lam,
        eval=True,
        sparsity=sparsity,
        m=m,
        pr=False
    )
    t_end = time.time()
    is_obj_runs.append([float(v) for v in IS.obj])
    is_w_runs.append(np.asarray(IS.w, dtype=float))
    is_time_runs.append(t_end - t_start)

    tr = Trust_Region(w0=w0.copy(), lam=lam, maxIter=maxIter, tol=tol)
    t_start = time.time()
    tr.fit(
        X_train=X,
        y_train=y,
        batch_size=batch_size,
        eval=True,
        sparsity=sparsity,
        pr=False,
        eta1=1e-3,
        eta2=1e-3,
        delta0=1.0,
        deltamax=10.0,
        gamma=2.0
    )
    t_end = time.time()
    tr_obj_runs.append([float(v) for v in tr.obj])
    tr_w_runs.append(np.asarray(tr.w, dtype=float))
    tr_time_runs.append(t_end - t_start)


scores_IS = average_obj_list(is_obj_runs)
w_IS = average_w_list(is_w_runs)

scores_tr = average_obj_list(tr_obj_runs)
w_tr = average_w_list(tr_w_runs)


results = {
    "important_sampling": {
        "obj": [float(v) for v in scores_IS],
        "w": np.asarray(w_IS).tolist(),
        "times": [float(v) for v in is_time_runs],
        "avg_time": float(np.mean(is_time_runs)),
        "sampling": {
            "type": "importance",
            "p": [float(v) for v in IS.p_sample.tolist()]
        }
    },
    "Trust_Region": {
        "obj": [float(v) for v in scores_tr],
        "w": np.asarray(w_tr).tolist(),
        "times": [float(v) for v in tr_time_runs],
        "avg_time": float(np.mean(tr_time_runs)),
    },
    "config": {
        "eta": float(eta),
        "lam": float(lam),
        "maxIter": int(maxIter),
        "tol": None if tol is None else float(tol),
        "batch_size": int(batch_size),
        "L_X": float(L_X),
        "sparsity": None if sparsity is None else int(sparsity),
        "m": int(m),
        "n_runs": int(n_runs),
    }
}

os.makedirs("./results", exist_ok=True)
result_path = "./results/CSI300_copy3.json"
with open(result_path, "w") as f:
    json.dump(results, f)

print(f"Results saved to {result_path}")


lw_map = {
    "improtance_sampling_SARAH": 2.0,
    "Trust_Region": 2.0
}

ls_map = {
    "improtance_sampling_SARAH": "-",
    "Trust_Region": "-"
}

label_map = {
    "Trust_Region": "PIHT",
    "improtance_sampling_SARAH": "VR-IHT-IS"
}


concatX = np.concatenate([X_train, X_test], axis=0)
concaty = np.concatenate([y_train, y_test], axis=0)

mydates = matplotlib.dates.num2date(
    matplotlib.dates.datestr2num(dataset.index)
)

plt.figure()

plt.plot(
    mydates,
    concaty,
    color="#E67700",
    label="CSI300 index",
    linewidth=2.0,
    linestyle="-"
)

plt.plot(
    mydates,
    scaler_X.transform(concatX) @ w_tr + y_offset,
    color="#d62728",
    label=label_map["Trust_Region"],
    linewidth=lw_map["Trust_Region"],
    linestyle=ls_map["Trust_Region"]
)

plt.plot(
    mydates,
    scaler_X.transform(concatX) @ w_IS + y_offset,
    color="#2ca02c",
    label=label_map["improtance_sampling_SARAH"],
    linewidth=lw_map["improtance_sampling_SARAH"],
    linestyle=ls_map["improtance_sampling_SARAH"]
)

plt.axvline(mydates[X_train.shape[0]], color="grey", linewidth=1.5)
plt.axvspan(mydates[X_train.shape[0]], mydates[-1], alpha=0.2, color="grey")

plt.xlabel("Date")
plt.ylabel("Index Value")
plt.legend()

locator = AutoDateLocator()
ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator))
plt.xticks(rotation=45)

plt.tight_layout()

os.makedirs("./plots", exist_ok=True)
plt.savefig("./plots/CSI300_copy3.png", dpi=200, bbox_inches="tight")
plt.show()


test_dates = mydates[X_train.shape[0]:]

plt.figure()

plt.plot(
    test_dates,
    y_test,
    color="#E67700",
    label="CSI300 index (test)",
    linewidth=2.0,
    linestyle="-"
)

plt.plot(
    test_dates,
    scaler_X.transform(X_test) @ w_tr + y_offset,
    color="#d62728",
    label=label_map["Trust_Region"],
    linewidth=lw_map["Trust_Region"],
    linestyle=ls_map["Trust_Region"]
)

plt.plot(
    test_dates,
    scaler_X.transform(X_test) @ w_IS + y_offset,
    color="#2ca02c",
    label=label_map["improtance_sampling_SARAH"],
    linewidth=lw_map["improtance_sampling_SARAH"],
    linestyle=ls_map["improtance_sampling_SARAH"]
)

plt.legend()

locator = AutoDateLocator()
ax = plt.gca()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator))
plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig("./plots/CSI300_test_only_copy3.png", dpi=200, bbox_inches="tight")
plt.show()

