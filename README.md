# Sample-Path Convergence of Probabilistic Variance-Reduced Iterative Hard Thresholding under a Stochastic Kurdyka–Łojasiewicz Framework

This repository contains the source code and experimental scripts for the **VR-IHT-IS** framework, which addresses finite-sum optimization problems with an explicit $\ell_0$ constraint.

The repository implements and evaluates the following methods:

- **PIHT**: Probabilistic Iterative Hard Thresholding.
- **VR-IHT-IS**: Mini-Batch VR-IHT with static importance sampling.
- **VR-IHT**: Mini-Batch VR-IHT with uniform sampling.

## Repository Structure

```text
.
├── SARAH_IHT1-support/
│   ├── importance_sampling.py
│   ├── run_support_experiment.py
│   └── plot_support_experiment.py
├── SARAH-IHT2-heterogeneity/
│   ├── heterogeneous.py
│   ├── importance_sampling.py
│   ├── uniform_sampling.py
│   ├── run_experiment.py
│   └── plot_results.py
├── SARAH_IHT3/
│   ├── importance_sampling.py
│   ├── Trust_Region.py
│   ├── run_experiment.py
│   └── plot_results.py
├── SARAH_IHT4/
│   ├── importance_sampling.py
│   ├── Trust_Region.py
│   ├── CSI300.py
│   ├── HSI.py
│   ├── SP500.py
│   └── plot_results.py
└── SARAH_IHT5/
    ├── importance_sampling.py
    ├── Trust_Region.py
    ├── dna_run_experiment.py
    ├── dna_plot_results.py
    ├── mnist_run_experiment.py
    └── mnist_plot_results.py
```

Each experiment directory contains the algorithm implementations, experiment drivers, input data when applicable, and directories for generated results and figures.

## Methods

### PIHT

PIHT uses probabilistic mini-batch updates combined with iterative hard thresholding. The trust-region implementation is provided in `Trust_Region.py` and is kept separate from the sampling-based implementations.

### VR-IHT-IS

VR-IHT-IS uses a SARAH-type variance-reduced gradient estimator with static importance probabilities. The sampling distribution is determined before optimization and is based on sample-wise smoothness or Lipschitz upper bounds:

\[
p_i = \frac{L_i}{\sum_{j=1}^{N} L_j}.
\]

The corresponding importance correction uses $1/(N p_i)$.

### VR-IHT

VR-IHT uses the same mini-batch variance-reduction framework with uniform sampling. Each mini-batch is sampled uniformly with replacement, without using sample-specific importance weights.

## Experiments

### Experiment 1: Support Identification

Directory: `SARAH_IHT1-support/`

This experiment studies finite support identification under importance sampling. It records support-related quantities such as:

- cumulative support changes;
- the symmetric difference between the estimated and true supports;
- the threshold gap; and
- the support stabilisation iteration.

Run the experiment from its directory:

```powershell
cd SARAH_IHT1-support
python run_support_experiment.py
python plot_support_experiment.py
```

Results are written to `results/`, and figures are written to `plots/`.

### Experiment 2: Heterogeneous Smoothness

Directory: `SARAH-IHT2-heterogeneity/`

This experiment evaluates VR-IHT and VR-IHT-IS on synthetic sparse regression problems with heterogeneous sample-wise Lipschitz constants. The heterogeneity level is controlled by the parameter `sigma`.

```powershell
cd SARAH-IHT2-heterogeneity
python run_experiment.py
python plot_results.py
```

The generated JSON file is stored in `results/heterogeneous_experiments.json`, and the figures are stored in `figures/`.

### Experiment 3: Sparse Classification

Directory: `SARAH_IHT3/`

This experiment applies importance-sampled VR-IHT and PIHT to sparse binary classification. The runner reads a LIBSVM-format dataset.

```powershell
cd SARAH_IHT3
python run_experiment.py
python plot_results.py --dataset w8a --loss sigmoid
```

The exact dataset and loss options depend on the data files and result files available in the directory. Use the command-line help to inspect the supported arguments:

```powershell
python run_experiment.py --help
python plot_results.py --help
```

### Experiment 4: Financial Time-Series Prediction

Directory: `SARAH_IHT4/`

This experiment evaluates importance-sampled VR-IHT and PIHT on financial time-series datasets, including CSI 300, HSI, and S\&P 500 data.

Available experiment drivers include:

```powershell
cd SARAH_IHT4
python CSI300.py
python HSI.py
python SP500.py
python plot_results.py
```

The drivers save numerical results under `results/` and generated figures under `plots/`.

### Experiment 5: Multiclass Classification

Directory: `SARAH_IHT5/`

This experiment evaluates importance-sampled VR-IHT and PIHT for multiclass classification using one-hot encoded labels. 

For the DNA dataset:

```powershell
cd SARAH_IHT5
python dna_run_experiment.py 0.2
python dna_plot_results.py 0.2
```

For the MNIST dataset:

```powershell
python mnist_run_experiment.py 0.2
python mnist_plot_results.py 0.2
```

The positional sparsity argument is interpreted as a ratio and converted internally to an integer sparsity level. Results are stored in `results/`.

## Installation

The code is written in Python and uses standard scientific-computing and machine-learning packages:

- NumPy
- SciPy
- scikit-learn
- Matplotlib
- joblib
- libsvm, where required by the LIBSVM data readers

Create and activate a virtual environment, then install the required packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-learn matplotlib joblib libsvm-official
```

On systems where `libsvm-official` is unavailable, install the LIBSVM package required by the local data-loading scripts.

## Reproducibility

The experiment scripts use fixed seeds or seed sequences for repeated runs. Main configuration parameters, including the sample size, dimension, sparsity level, regularization coefficient, mini-batch size, number of iterations, and number of trials, are defined near the beginning of the corresponding experiment scripts.

For reproducible results:

1. use the same Python and package versions;
2. run each script from its own experiment directory;
3. keep the supplied input data unchanged; and
4. avoid changing random seeds or experiment configuration parameters.

## Outputs

Depending on the experiment, the scripts generate:

- objective-function trajectories;
- elapsed-time curves;
- test accuracy measurements;
- final model parameters;
- effective gradient-computation counts;
- support-identification statistics; and
- PNG figures for visual comparison.

Generated files are stored in the `results/`, `plots/`, or `figures/` directory of the corresponding experiment.

## Notes

- `importance_sampling.py` contains the importance-sampling implementation in each experiment directory.
- `uniform_sampling.py` contains the uniform-sampling implementation for the heterogeneous-smoothness experiment.
- `Trust_Region.py` contains the PIHT implementation and is kept separate from the sampling-based methods.
- Existing result files are not overwritten unless the corresponding experiment script is run again.

## Contact

**Xiangyu Yang**  
School of Mathematics and Statistics  
Henan University  
Kaifeng 475000, Henan, China  
Email: [yangxy@henu.edu.cn](mailto:yangxy@henu.edu.cn)
