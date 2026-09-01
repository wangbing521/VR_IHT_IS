import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

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


def plot_all_results(
    json_path,
    dataset,
    loss,
    plot_dir="./plots",
    T_plot=1200
):
    os.makedirs(plot_dir, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plot_order = ["Importance_Sampling", "Trust_Region"]

    color_map = {
        "Importance_Sampling": "#2ca02c",
        "Trust_Region": "#d62728"
    }

    lw_map = {
        "Importance_Sampling": 2.4,
        "Trust_Region": 2.4
    }

    ls_map = {
        "Importance_Sampling": "-",
        "Trust_Region": "-"
    }

    label_map = {
        "Trust_Region": "PIHT",
        "Importance_Sampling": "VR-IHT-IS"
    }


    avg_obj_curve = {}

    for method in data.keys():
        if "obj" not in data[method]:
            continue

        obj_runs = data[method]["obj"]
        valid_runs = [np.asarray(r, dtype=float) for r in obj_runs if len(r) > 0]
        if not valid_runs:
            continue

        T = min(len(r) for r in valid_runs)
        if T == 0:
            continue

        arr = np.array([r[:T] for r in valid_runs], dtype=float)
        avg_obj_curve[method] = arr.mean(axis=0)

    if not avg_obj_curve:
        print("avg_obj_curve is empty; objective cannot be plotted.")
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))

        for method in plot_order:
            if method not in avg_obj_curve:
                continue

            mean_F = avg_obj_curve[method]
            y = mean_F[:T_plot]
            x = np.arange(len(y))

            ax.plot(
                x, y,
                label=label_map.get(method, method),
                color=color_map.get(method, None),
                linewidth=lw_map.get(method, 2.8),
                linestyle=ls_map.get(method, "-")
            )

        def _nice_tick_step(T):
            raw = max(1, T // 10)
            magnitude = 10 ** (len(str(raw)) - 1)
            if raw <= magnitude:
                return magnitude
            elif raw <= 2 * magnitude:
                return 2 * magnitude
            elif raw <= 5 * magnitude:
                return 5 * magnitude
            else:
                return 10 * magnitude

        tick_step = _nice_tick_step(T_plot)
        xticks = np.arange(0, T_plot + 1, tick_step)
        ax.set_xticks(xticks)

        ax.set_xlabel("Iterations", fontsize=16)
        ax.set_ylabel("Objective function value", fontsize=16)
        ax.set_xlim(0, T_plot)
        ax.legend(loc="upper right", frameon=True, fontsize=13, framealpha=0.9)


        plt.tight_layout()

        save_path = f"{plot_dir}/{dataset}+{loss}_iter_0.3.png"
        plt.savefig(save_path, dpi=350, bbox_inches="tight")
        plt.close()
        print("[OK] saved plot:", save_path)

    avg_obj_curve = {}
    avg_time_curve = {}

    for method in data.keys():
        if "obj" not in data[method] or "time" not in data[method]:
            continue

        obj_runs = data[method]["obj"]
        time_runs = data[method]["time"]

        valid_pairs = []
        for obj_r, time_r in zip(obj_runs, time_runs):
            if len(obj_r) > 0 and len(time_r) > 0:
                obj_r = np.asarray(obj_r, dtype=float)
                time_r = np.asarray(time_r, dtype=float)

                T = min(len(obj_r), len(time_r))
                if T > 0:
                    valid_pairs.append((obj_r[:T], time_r[:T]))

        if not valid_pairs:
            continue

        T_common = min(T_plot, min(len(obj_r) for obj_r, _ in valid_pairs))
        if T_common == 0:
            continue

        obj_arr = np.array([obj_r[:T_common] for obj_r, _ in valid_pairs], dtype=float)
        time_arr = np.array([time_r[:T_common] for _, time_r in valid_pairs], dtype=float)

        avg_obj_curve[method] = obj_arr.mean(axis=0)
        avg_time_curve[method] = time_arr.mean(axis=0)

    if not avg_obj_curve:
        print("avg_obj_curve is empty; objective versus time cannot be plotted.")
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))

        for method in plot_order:
            if method not in avg_obj_curve or method not in avg_time_curve:
                continue

            mean_F = avg_obj_curve[method]
            mean_time = avg_time_curve[method]

            ax.plot(
                mean_time, mean_F,
                label=label_map.get(method, method),
                color=color_map.get(method, None),
                linewidth=lw_map.get(method, 2.8),
                linestyle=ls_map.get(method, "-")
            )

        ax.set_xlabel("Elapsed time (s)", fontsize=16)
        ax.set_ylabel("Objective function value", fontsize=16)
        ax.legend(loc="upper right", frameon=True, fontsize=13, framealpha=0.9)
        ax.set_xlim(left=0)

        plt.tight_layout()

        save_path = f"{plot_dir}/{dataset}+{loss}_time_0.3.png"
        plt.savefig(save_path, dpi=350, bbox_inches="tight")
        plt.close()
        print("[OK] saved plot:", save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results directly from saved json.")
    parser.add_argument("--dataset", type=str, default="w8a", help="dataset name, e.g. a9a")
    parser.add_argument("--loss", type=str, default="sigmoid", help="loss name")
    parser.add_argument("--json_path", type=str, default=None, help="path to saved json")
    parser.add_argument("--plot_dir", type=str, default=None, help="directory to save plots")
    parser.add_argument("--T_plot", type=int, default=1200, help="max iterations to plot")

    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if args.json_path is None:
        json_path = os.path.join(BASE_DIR, 'results', f"{args.dataset}{args.loss}_0.3.json")
    else:
        json_path = args.json_path

    if args.plot_dir is None:
        plot_dir = os.path.join(BASE_DIR, 'plots')
    else:
        plot_dir = args.plot_dir

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"json file not found: {json_path}")

    print("[INFO] loading json:", json_path)

    plot_all_results(
        json_path=json_path,
        dataset=args.dataset,
        loss=args.loss,
        plot_dir=plot_dir,
        T_plot=args.T_plot
    )
