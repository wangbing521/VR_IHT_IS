import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from sklearn.preprocessing import StandardScaler
from matplotlib.dates import AutoDateFormatter, AutoDateLocator

from data import load_hsi, load_csi300, load_sp500

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

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
    "figure.figsize": (8, 4.8),
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

TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
LEGEND_FONTSIZE = 12

with open(os.path.join(RESULTS_DIR, "HSI_copy.json"), "r") as f:
    results = json.load(f)

w_is = np.array(results["important_sampling"]["w"])
w_tr = np.array(results["Trust_Region"]["w"])

dataset = load_hsi(download=False)
npdata = dataset.values

split = int(0.8 * npdata.shape[0])
data_train, data_test = npdata[:split, :], npdata[split:, :]

X_train, y_train = data_train[:, :-1], data_train[:, -1]
X_test, y_test = data_test[:, :-1], data_test[:, -1]

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_offset = np.mean(y_train)

concatX = np.concatenate([X_train, X_test], axis=0)
concaty = np.concatenate([y_train, y_test], axis=0)
concatX_scaled = scaler_X.transform(concatX)

mydates = matplotlib.dates.num2date(matplotlib.dates.datestr2num(dataset.index))
test_dates = mydates[X_train.shape[0]:]

os.makedirs(PLOTS_DIR, exist_ok=True)

fig, ax = plt.subplots()

ax.plot(
    mydates,
    concaty,
    color="#E67700",
    alpha=0.7,
    linewidth=2.0,
    label="HSI index"
)

ax.plot(
    mydates,
    concatX_scaled @ w_tr + y_offset,
    color="#d62728",
    alpha=0.85,
    linewidth=2.0,
    label="PIHT"
)

ax.plot(
    mydates,
    concatX_scaled @ w_is + y_offset,
    color="#2ca02c",
    alpha=0.85,
    linewidth=2.0,
    label="VR-IHT-IS"
)

ax.axvline(mydates[X_train.shape[0]], color="grey")
ax.axvspan(mydates[X_train.shape[0]], mydates[-1], alpha=0.2, color="grey")

ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Index Value", fontsize=LABEL_FONTSIZE)
ax.legend(fontsize=LEGEND_FONTSIZE)

locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator))

ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=45)
ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "HSI.png"), dpi=300, bbox_inches="tight")
plt.show()

ax.plot(
    test_dates,
    y_test,
    color="#E67700",
    alpha=0.7,
    linewidth=2.0,
    label="HSI index (test)"
)

ax.plot(
    test_dates,
    X_test_scaled @ w_tr + y_offset,
    color="#d62728",
    alpha=0.85,
    linewidth=2.0,
    label="PIHT"
)

ax.plot(
    test_dates,
    X_test_scaled @ w_is + y_offset,
    color="#2ca02c",
    alpha=0.85,
    linewidth=2.0,
    label="VR-IHT-IS"
)

ax.set_xlabel("Date", fontsize=LABEL_FONTSIZE)
ax.set_ylabel("Index Value", fontsize=LABEL_FONTSIZE)
ax.legend(fontsize=LEGEND_FONTSIZE)


locator = AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(AutoDateFormatter(locator))

ax.tick_params(axis="x", labelsize=TICK_FONTSIZE, rotation=45)
ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "HSI_test_only.png"), dpi=300, bbox_inches="tight")
plt.show()
