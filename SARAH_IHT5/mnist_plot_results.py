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
    sparsity_ratio = float(sys.argv[1])
    tag = f"sparsity{str(sparsity_ratio).replace('.', '_')}"

    LABEL_FONTSIZE = 14
    TITLE_FONTSIZE = 14
    LEGEND_FONTSIZE = 12
    TICK_FONTSIZE = 12

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(BASE_DIR, "results")

    tr_file = os.path.join(results_dir, f"results_TR_{tag}_mnist.json")
    is_file = os.path.join(results_dir, f"results_improtant_sample_{tag}_mnist.json")

    if not os.path.exists(tr_file):
        print(f"WARNING: {tr_file} not found, only plotting IS")
        results_TR = None
    else:
        with open(tr_file, "r", encoding="utf-8") as f:
            results_TR = json.load(f)

    if not os.path.exists(is_file):
        print(f"WARNING: {is_file} not found, only plotting TR")
        results_improtant_sample = None
    else:
        with open(is_file, "r", encoding="utf-8") as f:
            results_improtant_sample = json.load(f)

    if results_TR is None and results_improtant_sample is None:
        print("ERROR: No result files found. Exiting.")
        sys.exit(1)

    color_map = {
        "improtant_sample": "#2ca02c",
        "Trust_Region": "#d62728"
    }

    label_map = {
        "Trust_Region": "PIHT",
        "improtant_sample": "VR-IHT-IS"
    }

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    if results_improtant_sample is not None:
        ax.plot(
            results_improtant_sample["iter"], results_improtant_sample["losses"],
            color=color_map["improtant_sample"],
            label=label_map["improtant_sample"],
            linewidth=2.0, alpha=0.85
        )

    if results_TR is not None:
        ax.plot(
            results_TR["iter"], results_TR["losses"],
            color=color_map["Trust_Region"],
            label=label_map["Trust_Region"],
            linewidth=2.0, alpha=0.85
        )

    ax.set_xlabel("Iterations", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Objective function value", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)

    out1 = os.path.join(results_dir, f"mnist_iter_loss_{tag}.png")
    fig.savefig(out1, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))

    if results_improtant_sample is not None:
        ax.plot(
            results_improtant_sample["time"], results_improtant_sample["losses"],
            color=color_map["improtant_sample"],
            label=label_map["improtant_sample"],
            linewidth=2.0, alpha=0.85
        )

    if results_TR is not None:
        ax.plot(
            results_TR["time"], results_TR["losses"],
            color=color_map["Trust_Region"],
            label=label_map["Trust_Region"],
            linewidth=2.0, alpha=0.85
        )

    ax.set_xlabel("Elapsed time (s)", fontsize=LABEL_FONTSIZE)
    ax.set_ylabel("Objective function value", fontsize=LABEL_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)

    out2 = os.path.join(results_dir, f"mnist_time_loss_{tag}.png")
    fig.savefig(out2, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")

    print("Done.")
