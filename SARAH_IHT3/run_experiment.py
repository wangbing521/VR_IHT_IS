from importance_sampling import Importance_Sampling
from Trust_Region import Trust_Region

import numpy as np
from sklearn.preprocessing import normalize
import json
from joblib import Memory
from sklearn.datasets import load_svmlight_file
import argparse
import os

mem = Memory("./mycache")


@mem.cache
def get_data(filename):
    data = load_svmlight_file(filename)
    return data[1], data[0]


def to_python(obj):
    """
    Recursively convert NumPy objects to native Python types for JSON serialization.
    """
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_python(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj


def as_serializable(x):
    """
    Convert model outputs to JSON-serializable Python types.
    Scalars -> int/float
    Arrays/lists -> list
    """
    return np.asarray(x).tolist()


methods = ['Importance_Sampling', 'Trust_Region']

flags = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description="SARAH-IHT VS Trust_Region VS Importance_Sampling"
)

flags.add_argument('--loss', type=str, default='sigmoid', help='Type of loss')
flags.add_argument('--typedata', type=str, default='small', help='Type of data')
flags.add_argument('--iter', type=int, default=3000, help='Number of iters')
flags.add_argument('--run', type=int, default=20, help='Number of runs')
flags.add_argument('--sparsity', type=float, default=0.3, help='Sparsity ratio (0.1 / 0.2 / 0.3 / 0.4 / 0.6)')

FLAGS = flags.parse_args()

loss = FLAGS.loss

if FLAGS.typedata == 'small':
    datasets = ['a9a']

os.makedirs('./results', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

plot_order = ["improtance_sampling_SARAH", "Trust_Region"]
plot_order_small = ["improtance_sampling_SARAH", "Trust_Region"]

color_map = {
    "improtance_sampling_SARAH": "blue",
    "Trust_Region": "red"
}

lw_map = {
    "improtance_sampling_SARAH": 2.0,
    "Trust_Region": 2.0
}

ls_map = {
    "improtance_sampling_SARAH": "-",
    "Trust_Region": "-"
}

for dataset in datasets:
    path = './data/'

    if FLAGS.typedata == 'small':
        path += dataset + '.txt'
    else:
        path += dataset

    y_all, X_all = get_data(path)
    print('check read data')

    maxiter = FLAGS.iter
    lam = 1e-3
    tol = 1e-4
    eta = 4

    if loss == 'sigmoid':
        ll = 1 / (6 * np.sqrt(3))
    elif loss == 'NN2':
        ll = (39 + 55 * np.sqrt(33)) / 2304
    else:
        raise ValueError(f"Unknown loss: {loss}")

    print('check', X_all.shape)

    print("preprocesing data")

    data = {}
    for method in methods:
        data[method] = {}
        data[method]['obj'] = []
        data[method]['time'] = []
        data[method]['test_acc'] = []
        data[method]['iter'] = []
        data[method]['final_obj'] = []

    for iterrun in range(FLAGS.run):
        seed_base = 42
        np.random.seed(seed_base + iterrun)

        test_ratio = 0.2
        total_size = X_all.shape[0]

        test_size = int(total_size * test_ratio)
        train_size = total_size - test_size

        rnd_indices = np.random.permutation(total_size)

        X_train_run = X_all[rnd_indices[:train_size]].copy()
        y_train_run = np.c_[y_all][rnd_indices[:train_size]]

        X_test_run = X_all[rnd_indices[-test_size:]].copy()
        y_test_run = np.c_[y_all][rnd_indices[-test_size:]]

        m_train = X_train_run.shape[0]
        n_feat = X_train_run.shape[1]

        X_train_run_norm = X_train_run.copy()
        X_test_run_norm = X_test_run.copy()

        X_train_run_norm = normalize(X_train_run_norm, 'l2', axis=1, copy=False)
        X_test_run_norm = normalize(X_test_run_norm, 'l2', axis=1, copy=False)

        row_norm_sq = np.asarray(X_train_run_norm.multiply(X_train_run_norm).sum(axis=1)).ravel()
        l = float(np.max(row_norm_sq)) * ll + lam

        w0 = np.zeros((n_feat, 1))

        inner_loop = 2 * int(m_train ** (1 / 2))
        batch = int(m_train ** (1 / 2))
        sparsity = int(FLAGS.sparsity * n_feat)


        method = 'Importance_Sampling'
        print('run', iterrun, 'method', method)

        model = Importance_Sampling(
            w0,
            eta=eta,
            lam=lam,
            maxIter=maxiter,
            tol=tol
        )

        model.fit(
            X_train_run,
            X_train_run_norm,
            y_train_run,
            X_test_run_norm,
            y_test_run,
            batch_size=batch,
            L=l,
            loss=loss,
            eval=True,
            sparsity=sparsity,
            m=inner_loop,
            pr=False
        )

        data[method]['obj'].append(as_serializable(model.obj))
        data[method]['time'].append(as_serializable(model.Time))
        data[method]['test_acc'].append(as_serializable(model.acc_test))
        data[method]['iter'].append(as_serializable(model.iter))
        data[method]['final_obj'].append(as_serializable(model.obj[-1]))


        method = 'Trust_Region'
        print('run', iterrun, 'method', method)

        model = Trust_Region(
            w0,
            lam=lam,
            maxIter=maxiter,
            tol=tol
        )

        model.fit(
            X_train_run_norm,
            y_train_run,
            X_test_run_norm,
            y_test_run,
            batch_size=batch,
            loss=loss,
            eval=True,
            sparsity=sparsity,
            pr=False,
            eta1=1e-3,
            eta2=1e-3,
            delta0=1.0,
            deltamax=10.0,
            gamma=2.0
        )

        data[method]['obj'].append(as_serializable(model.obj))
        data[method]['time'].append(as_serializable(model.Time))
        data[method]['test_acc'].append(as_serializable(model.acc_test))
        data[method]['iter'].append(as_serializable(model.iter))
        data[method]['final_obj'].append(as_serializable(model.obj[-1]))

    data = to_python(data)

    print(f"\n===== {dataset} final average test_acc =====")
    for method in data.keys():
        acc_runs = data[method]["test_acc"]
        valid_runs = [r for r in acc_runs if len(r) > 0]
        if not valid_runs:
            continue

        final_acc = np.mean([r[-1] for r in valid_runs])
        data[method]["avg_final_test_acc"] = np.asarray(final_acc).tolist()
        print(f"{method}: {final_acc:.6f}")

    for method in data.keys():
        obj_runs = data[method]["final_obj"]
        valid_runs = [np.asarray(obj, dtype=float) for obj in obj_runs]

        if not valid_runs:
            continue

        obj_avg = np.mean(np.stack(valid_runs, axis=0), axis=0)
        data[method]["avg_final_obj"] = np.asarray(obj_avg).tolist()
        print(f"{method}: {obj_avg:.6f}")
    os.makedirs('./results', exist_ok=True)
    json_path = os.path.join('.', 'results', f'{dataset}{loss}_{FLAGS.sparsity}.json')
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("[OK] saved json:", json_path)
