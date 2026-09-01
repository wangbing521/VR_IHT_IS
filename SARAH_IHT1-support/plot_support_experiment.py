import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

LABEL_FS = 14
TITLE_FS = 14
LEGEND_FS = 12
TICK_FS = 12

REGIME_STYLE = {
    "small":  {"color": "#2ca02c", "marker": "o", "label": r"$\alpha$=0.1"},
    "medium": {"color": "#d62728", "marker": "s", "label": r"$\alpha$=0.4"},
    "large":  {"color": "#1f77b4", "marker": "^", "label": r"$\alpha$=1.0"},
}

METHOD_LABEL = {"IS": "VR-IHT-IS"}
METHOD_COLOR = {"IS": "#2ca02c"}
BOX_COLOR = "#E67700"

REGIMES = ["small", "medium", "large"]
METHODS = ["IS"]

DOWNSAMPLE = 5


def load_results():
    """Load the combined results JSON."""
    filepath = os.path.join(RESULTS_DIR, "support_experiment_all.json")
    if not os.path.exists(filepath):
        all_results = {}
        for regime in REGIMES:
            for method in METHODS:
                key = f"{regime}_{method}"
                fp = os.path.join(RESULTS_DIR, f"support_{key}.json")
                if os.path.exists(fp):
                    with open(fp, "r", encoding="utf-8") as f:
                        all_results[key] = json.load(f)
        return all_results

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def downsample(arr, step):
    """Downsample array for cleaner plots."""
    if arr is None or len(arr) == 0:
        return np.array([])
    return np.asarray(arr)[::step]


def _plot_median_iqr(ax, results, field, regimes, log_y=False):
    for regime in regimes:
        key = f"{regime}_IS"
        if key not in results:
            continue
        r = results[key]
        iters = downsample(r["iter"], DOWNSAMPLE)
        med = np.asarray(downsample(r[field], DOWNSAMPLE), dtype=float)
        q1 = np.asarray(downsample(r.get(field + "_q1", []), DOWNSAMPLE), dtype=float)
        q3 = np.asarray(downsample(r.get(field + "_q3", []), DOWNSAMPLE), dtype=float)
        n = min(len(iters), len(med))
        x = np.asarray(iters[:n])
        y = med[:n]
        style = REGIME_STYLE[regime]
        if log_y:
            y = np.maximum(y, 1e-16)
            y_q1 = np.maximum(q1[:n], 1e-16) if len(q1) > 0 else None
            y_q3 = np.maximum(q3[:n], 1e-16) if len(q3) > 0 else None
            ax.semilogy(x, y, color=style["color"], linewidth=2.0, label=style["label"])
        else:
            y_q1 = q1[:n] if len(q1) > 0 else None
            y_q3 = q3[:n] if len(q3) > 0 else None
            ax.plot(x, y, color=style["color"], linewidth=2.0, label=style["label"])
        if y_q1 is not None and y_q3 is not None and len(y_q1) > 0:
            ax.fill_between(x, y_q1, y_q3, color=style["color"], alpha=0.15)


def plot_N_switch(results):
    """Staircase plot of cumulative support changes, median ± IQR."""
    fig, ax = plt.subplots()
    _plot_median_iqr(ax, results, "N_switch", REGIMES)
    for line in ax.get_lines():
        line.set_drawstyle("steps-post")
    ax.set_xlabel("Iterations", fontsize=LABEL_FS)
    ax.set_ylabel(r"$N_{\mathrm{switch}}(k)$", fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc="lower right")
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "N_switch_vs_iter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: N_switch_vs_iter.png")


def plot_M_nat(results):
    """Symmetric difference with true support, median ± IQR."""
    fig, ax = plt.subplots()
    _plot_median_iqr(ax, results, "M_nat", REGIMES)
    ax.set_xlabel("Iterations", fontsize=LABEL_FS)
    ax.set_ylabel(r"$M_{\mathrm{nat}}(k)$", fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc="upper right")
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "M_nat_vs_iter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: M_nat_vs_iter.png")


def plot_gamma_hat(results):
    """Threshold gap, median ± IQR, log scale."""
    fig, ax = plt.subplots()
    _plot_median_iqr(ax, results, "gamma_hat", REGIMES, log_y=True)
    ax.set_xlabel("Iterations", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\widehat{\gamma}_k$  (log scale)", fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc="lower right")
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "gamma_hat_vs_iter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: gamma_hat_vs_iter.png")


def plot_T_stab_boxplot(results):
    """Boxplot of support stabilisation times."""
    fig, ax = plt.subplots()

    data_is = []
    labels = []

    for regime in REGIMES:
        key_is = f"{regime}_IS"

        if key_is in results and "T_stab" in results[key_is]:
            data_is.append(results[key_is]["T_stab"])
        else:
            data_is.append([])

        labels.append(REGIME_STYLE[regime]["label"])

    n_regimes = len(REGIMES)
    x = np.arange(n_regimes)

    bp1 = ax.boxplot(data_is, positions=x, widths=0.5,
                     patch_artist=True,
                     boxprops=dict(facecolor="none", edgecolor=BOX_COLOR, linewidth=2.2),
                     medianprops=dict(color="black", linewidth=2.2),
                     whiskerprops=dict(color="black", linestyle="--", linewidth=2.2),
                     capprops=dict(color="black", linewidth=2.2))

    for i, d_is in enumerate(data_is):
        if len(d_is) > 0:
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(d_is))
            ax.scatter(np.full_like(d_is, x[i]) + jitter, d_is,
                       color=BOX_COLOR, alpha=0.4, s=20, edgecolors="none",
                       label="Individual trials" if i == 0 else None)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FS)
    ax.set_xlabel("Iterations", fontsize=LABEL_FS, color="none")
    ax.set_ylabel(r"$T_{\mathrm{stab}}$  (iterations)", fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS, loc="upper right")
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, axis="y")

    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "T_stab_boxplot.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: T_stab_boxplot.png")


def plot_loss(results):
    """Objective value, median ± IQR, log scale."""
    fig, ax = plt.subplots()
    _plot_median_iqr(ax, results, "obj", REGIMES, log_y=True)
    ax.set_xlabel("Iterations", fontsize=LABEL_FS)
    ax.set_ylabel(r"$F(x^k)$  (log scale)", fontsize=LABEL_FS)
    ax.legend(fontsize=LEGEND_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(True, axis="y")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "loss_vs_iter.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: loss_vs_iter.png")


def main():
    print("=" * 60)
    print("Plotting Support Identification Results")
    print("=" * 60)

    results = load_results()
    if not results:
        print("ERROR: No results found. Run 'run_support_experiment.py' first.")
        return

    print(f"Loaded {len(results)} result sets")

    plot_N_switch(results)
    plot_M_nat(results)
    plot_gamma_hat(results)
    plot_T_stab_boxplot(results)
    plot_loss(results)

    print(f"\nAll plots saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
