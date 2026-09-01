"""Plot results from the heterogeneous sparse linear regression experiment."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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

LABEL_FS = 14
LEGEND_FS = 12
TICK_FS = 12


def group_results(records):
    grouped = {}
    for record in records:
        sigma = record["sigma"]
        grouped.setdefault(sigma, {"VR_IHT": [], "VR_IHT_IS": []})
        for method in grouped[sigma]:
            grouped[sigma][method].append({
                "obj": np.asarray(record[method]["obj"]),
                "time": np.asarray(record[method]["time"]),
            })
    return grouped


def compute_mean_trajectories(trajectories):
    min_length = min(len(trajectory["obj"]) for trajectory in trajectories)
    aligned_objective = np.array([
        trajectory["obj"][:min_length] for trajectory in trajectories
    ])
    aligned_time = np.array([
        trajectory["time"][:min_length] for trajectory in trajectories
    ])
    mean_objective = np.mean(aligned_objective, axis=0)
    std_objective = np.std(aligned_objective, axis=0)
    mean_time = np.mean(aligned_time, axis=0)
    std_time = np.std(aligned_time, axis=0)
    return mean_objective, std_objective, mean_time, std_time


def plot_objective_trajectories(grouped, figures_dir):
    colors = {"VR_IHT": "#1f77b4", "VR_IHT_IS": "#2ca02c"}
    labels = {"VR_IHT": "VR-IHT (uniform)", "VR_IHT_IS": "VR-IHT-IS (importance)"}

    for sigma in sorted(grouped):
        figure, axis = plt.subplots()
        for method in ["VR_IHT", "VR_IHT_IS"]:
            trajectories = grouped[sigma][method]
            mean_objective, _, _, _ = compute_mean_trajectories(trajectories)
            axis.plot(
                np.arange(len(mean_objective)), mean_objective, color=colors[method],
                linewidth=2.0, label=labels[method],
            )

        axis.set_xlabel("Iterations", fontsize=LABEL_FS)
        axis.set_ylabel("Objective function value", fontsize=LABEL_FS)
        axis.ticklabel_format(style="plain", axis="y", useOffset=False)
        axis.tick_params(axis="both", labelsize=TICK_FS)
        axis.legend(fontsize=LEGEND_FS, framealpha=0.8)
        axis.grid(True, axis="y")
        figure.tight_layout()
        figure.savefig(
            os.path.join(figures_dir, f"heterogeneous_objective_sigma{sigma:.1f}.png"),
            dpi=300,
        )
        plt.close(figure)


def plot_objective_time_trajectories(grouped, figures_dir):
    colors = {"VR_IHT": "#1f77b4", "VR_IHT_IS": "#2ca02c"}
    labels = {"VR_IHT": "VR-IHT (uniform)", "VR_IHT_IS": "VR-IHT-IS (importance)"}

    for sigma in sorted(grouped):
        figure, axis = plt.subplots()
        for method in ["VR_IHT", "VR_IHT_IS"]:
            trajectories = grouped[sigma][method]
            mean_objective, _, mean_time, _ = compute_mean_trajectories(trajectories)
            axis.plot(
                mean_time, mean_objective, color=colors[method],
                linewidth=2.0, label=labels[method],
            )

        axis.set_xlabel("Elapsed time (s)", fontsize=LABEL_FS)
        axis.set_ylabel("Objective function value", fontsize=LABEL_FS)
        axis.ticklabel_format(style="plain", axis="y", useOffset=False)
        axis.tick_params(axis="both", labelsize=TICK_FS)
        axis.legend(fontsize=LEGEND_FS, framealpha=0.8)
        axis.grid(True, axis="y")
        figure.tight_layout()
        figure.savefig(
            os.path.join(figures_dir, f"heterogeneous_objective_time_sigma{sigma:.1f}.png"),
            dpi=300,
        )
        plt.close(figure)


def plot_final_objective_gap(grouped, figures_dir):
    final_values = {
        sigma: {
            method: [trajectory["obj"][-1] for trajectory in trajectories]
            for method, trajectories in methods.items()
        }
        for sigma, methods in grouped.items()
    }

    sigmas = sorted(final_values)
    figure, axis = plt.subplots()
    means = []
    for sigma in sigmas:
        differences = np.asarray(final_values[sigma]["VR_IHT"]) - np.asarray(
            final_values[sigma]["VR_IHT_IS"]
        )
        means.append(np.mean(differences))

    axis.plot(
        sigmas, means, color="#E67700", marker="D", markersize=9, linewidth=2.0,
        label=r"$\mathbb{E}[F_{\mathrm{unif}} - F_{\mathrm{IS}}]$",
    )
    axis.axhline(0, color="gray", linestyle="--", linewidth=1.0)
    axis.set_xlabel(r"$\sigma$", fontsize=LABEL_FS)
    axis.set_ylabel(r"$\mathbb{E}[F_{\mathrm{unif}} - F_{\mathrm{IS}}]$", fontsize=LABEL_FS)
    axis.tick_params(axis="both", labelsize=TICK_FS)
    axis.legend(fontsize=LEGEND_FS, framealpha=0.8)
    axis.grid(True, axis="y")
    figure.tight_layout()
    figure.savefig(os.path.join(figures_dir, "heterogeneous_gap.png"), dpi=300)
    plt.close(figure)


def main():
    root = os.path.dirname(__file__)
    results_path = os.path.join(root, "results", "heterogeneous_experiments.json")
    with open(results_path, "r", encoding="utf-8") as input_file:
        records = json.load(input_file)

    figures_dir = os.path.join(root, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    grouped = group_results(records)
    plot_objective_trajectories(grouped, figures_dir)
    plot_objective_time_trajectories(grouped, figures_dir)
    plot_final_objective_gap(grouped, figures_dir)


if __name__ == "__main__":
    main()