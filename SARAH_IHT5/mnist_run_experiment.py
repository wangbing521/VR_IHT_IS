import numpy as np
import os
import json
import sys
import random

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline

from Trust_Region import Trust_Region
from importance_sampling import Importance_Sampling


rs = np.random.RandomState(42)
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "data", "mnist.txt")
results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

N_RUNS = 20

print("Loading mnist.txt ...", flush=True)
X_sparse, labels = load_svmlight_file(data_path)
labels = labels.astype(int)
print(f"  X shape: {X_sparse.shape}, labels: {len(set(labels))}", flush=True)

X_train_sparse, X_test_sparse, labels_train_raw, labels_test_raw = train_test_split(
    X_sparse, labels, test_size=0.2, random_state=42, stratify=labels
)

X_train = X_train_sparse.toarray()
X_test = X_test_sparse.toarray()

processor = make_pipeline(StandardScaler(), VarianceThreshold())
X_train = processor.fit_transform(X_train)
X_test = processor.transform(X_test)

try:
    encoder = OneHotEncoder(sparse_output=False)
except TypeError:
    encoder = OneHotEncoder(sparse=False)

y_train = encoder.fit_transform(labels_train_raw.reshape(-1, 1))
y_test = encoder.transform(labels_test_raw.reshape(-1, 1))

n_classes = y_train.shape[1]
n_train = X_train.shape[0]
d = X_train.shape[1]
print(f"  n_train={n_train}, d={d}, n_classes={n_classes}", flush=True)

c_mat = 0.5 * (np.eye(n_classes) - 1 / n_classes * np.ones((n_classes, n_classes)))

LAM = 1e-3
MAX_ITER = 4500


def get_L(lam):
    return np.linalg.norm(
        np.kron(c_mat, X_train.T @ X_train) / n_train + 2 * lam * np.eye(n_classes * d), ord=2
    )


def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64, np.integer)):
        return int(obj)
    elif isinstance(obj, list):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, tuple):
        return [to_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    else:
        return obj


def save_json(results, filename):
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(to_serializable(results), f, ensure_ascii=False, indent=2)


def average_scalar_list(values):
    return float(np.mean([float(v) for v in values]))


def average_same_shape_arrays(values):
    arr = np.asarray(values, dtype=float)
    return np.mean(arr, axis=0)


def average_1d_variable_length(values):
    max_len = max(len(v) for v in values)
    arr = np.full((len(values), max_len), np.nan, dtype=float)
    for i, v in enumerate(values):
        v = np.asarray(v, dtype=float).reshape(-1)
        arr[i, :len(v)] = v
    return np.nanmean(arr, axis=0)


def average_field(values):
    first = values[0]
    if np.isscalar(first):
        return average_scalar_list(values)
    arrays = [np.asarray(v, dtype=float) for v in values]
    shapes = [a.shape for a in arrays]
    if all(s == shapes[0] for s in shapes):
        return average_same_shape_arrays(arrays)
    if all(a.ndim == 1 for a in arrays):
        return average_1d_variable_length(arrays)
    raise ValueError("Unsupported field shape for averaging.")


def average_run_results(run_results):
    return {
        "method": run_results[0]["method"],
        "W": average_field([r["W"] for r in run_results]),
        "losses": average_field([r["losses"] for r in run_results]),
        "time": average_field([r["time"] for r in run_results]),
        "iter": average_field([r["iter"] for r in run_results]),
        "test_accuracy": average_field([r["test_accuracy"] for r in run_results]),
    }


def run_TR_once(X_train, y_train, X_test, y_test, k=150, seed=None):
    m_train = X_train.shape[0]
    d = X_train.shape[1]
    n_classes = y_train.shape[1]
    W0 = np.zeros((d, n_classes))
    batch_size = int(m_train ** 0.5)

    model = Trust_Region(w0=W0, lam=LAM, maxIter=MAX_ITER, tol=1e-4)
    model.fit(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        batch_size=batch_size, eval=True, sparsity=k, pr=False,
        eta1=1e-3, eta2=1e-3, delta0=1.0, deltamax=10.0, gamma=2.0
    )
    return {
        "method": "Trust_Region",
        "W": model.w, "losses": model.obj, "time": model.Time,
        "iter": model.iter, "test_accuracy": model.test_acc
    }


def run_improtant_sample_once(X_train, y_train, X_test, y_test, k=150, seed=None):
    d = X_train.shape[1]
    n_classes = y_train.shape[1]
    m_train = X_train.shape[0]
    L_is = get_L(LAM)
    W0 = np.zeros((d, n_classes))
    batch_size = int(m_train ** 0.5)
    inner_loop = 2 * int(m_train ** 0.5)

    model = Importance_Sampling(
        w0=W0, eta=4.0, lam=LAM, maxIter=MAX_ITER, tol=1e-4, c_mat=c_mat, seed=seed
    )
    model.fit(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
        batch_size=batch_size, L=L_is, eval=True, sparsity=k, m=inner_loop, pr=False
    )
    return {
        "method": "improtant_sample",
        "W": model.w, "losses": model.obj, "time": model.Time,
        "iter": model.iter, "test_accuracy": model.test_acc
    }


def run_multiple_times(run_once_fn, X_train, y_train, X_test, y_test, k=150, n_runs=20):
    run_results = []
    method_name = run_once_fn.__name__.replace("run_", "").replace("_once", "")
    for run_id in range(n_runs):
        seed = 42 + run_id
        random.seed(seed)
        np.random.seed(seed)
        print(f"  [{method_name}] run {run_id + 1}/{n_runs} (seed={seed})", flush=True)
        result = run_once_fn(X_train, y_train, X_test, y_test, k, seed=seed)
        run_results.append(result)
    return average_run_results(run_results)


def run_TR(X_train, y_train, X_test, y_test, k=150, n_runs=20):
    return run_multiple_times(run_TR_once, X_train, y_train, X_test, y_test, k, n_runs)


def run_improtant_sample(X_train, y_train, X_test, y_test, k=150, n_runs=20):
    return run_multiple_times(run_improtant_sample_once, X_train, y_train, X_test, y_test, k, n_runs)


if __name__ == "__main__":
    sparsity_ratio = float(sys.argv[1])
    k = int(sparsity_ratio * d * n_classes)
    tag = f"sparsity{str(sparsity_ratio).replace('.', '_')}"
    skip_tr = len(sys.argv) > 2 and sys.argv[2] == "--skip-tr"

    print(f"\nDataset: mnist  |  sparsity={sparsity_ratio}  |  k={k}")
    print(f"d={d}, n_classes={n_classes}, n_runs={N_RUNS}, max_iter={MAX_ITER}\n", flush=True)

    if not skip_tr:
        print(f">>> [1/2] Trust_Region ({N_RUNS} runs)")
        results_TR = run_TR(X_train, y_train, X_test, y_test, k, n_runs=N_RUNS)
        save_json(results_TR, f"results_TR_{tag}_mnist.json")
        print(f"  -> saved: results_TR_{tag}_mnist.json\n", flush=True)
    else:
        print(f">>> [1/2] Trust_Region -- SKIPPED\n")

    print(f">>> [2/2] improtant_sample ({N_RUNS} runs)")
    results_is = run_improtant_sample(X_train, y_train, X_test, y_test, k, n_runs=N_RUNS)
    save_json(results_is, f"results_improtant_sample_{tag}_mnist.json")
    print(f"  -> saved: results_improtant_sample_{tag}_mnist.json\n", flush=True)

    print("All done.")
