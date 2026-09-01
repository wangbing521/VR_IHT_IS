import matplotlib.pyplot as plt
import json
import sys
import os

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

if __name__ == "__main__":
    k = int(sys.argv[1])

    LABEL_FONTSIZE = 14
    TITLE_FONTSIZE = 14
    LEGEND_FONTSIZE = 12
    TICK_FONTSIZE = 12

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(BASE_DIR, "results")

    with open(os.path.join(results_dir, f"results_TR_k_{k}_now3.json"), "r", encoding="utf-8") as file:
        results_TR = json.load(file)

    with open(os.path.join(results_dir, f"results_improtant_sample_k_{k}_now3.json"), "r", encoding="utf-8") as file:
        results_improtant_sample = json.load(file)

    lw_map = {
        "improtant_sample": 2.0,
        "Trust_Region": 2.0
    }

    ls_map = {
        "improtant_sample": "-",
        "Trust_Region": "-"
    }

    color_map = {
        "improtant_sample": "#2ca02c",
        "Trust_Region": "#d62728"
    }

    label_map = {
        "Trust_Region": "PIHT",
        "improtant_sample": "VR-IHT-IS"
    }

    results_map = {
        "Trust_Region": results_TR,
        "improtant_sample": results_improtant_sample
    }

    plt.figure()

    for method in ["Trust_Region", "improtant_sample"]:
        results = results_map[method]
        plt.plot(
            results["iter"],
            results["losses"],
            color=color_map[method],
            label=label_map[method],
            linewidth=lw_map[method],
            linestyle=ls_map[method],
            markersize=5,
            alpha=0.8
        )

    plt.xlabel("Iterations", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Objective function value", fontsize=LABEL_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)

    plt.savefig(
        os.path.join(results_dir, f"improtant_sample_iter_loss_k_{k}_now3.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    plt.figure()

    for method in ["Trust_Region", "improtant_sample"]:
        results = results_map[method]
        plt.plot(
            results["time"],
            results["losses"],
            color=color_map[method],
            label=label_map[method],
            linewidth=lw_map[method],
            linestyle=ls_map[method],
            markersize=5
        )

    plt.xlabel("Elapsed time (s)", fontsize=LABEL_FONTSIZE)
    plt.ylabel("Objective function value", fontsize=LABEL_FONTSIZE)
    plt.legend(fontsize=LEGEND_FONTSIZE)
    plt.xticks(fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)

    plt.savefig(
        os.path.join(results_dir, f"improtant_sample_time_loss_k_{k}_now3.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
