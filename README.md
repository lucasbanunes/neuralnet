# neuralnet

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Keras 3](https://img.shields.io/badge/Keras-3-red.svg)](https://keras.io/)
[![Polars](https://img.shields.io/badge/Polars-1.x-blueviolet.svg)](https://pola.rs/)

**neuralnet** is a high-performance deep learning and quantization framework developed for the **Neural Ringer Algorithm** deployed in the Level 0 (L0) and Level 2 (L2) calorimeter trigger electron identification at the **ATLAS Experiment (CERN)**.

---

## Table of Contents

- [Overview & Physics Context](#overview--physics-context)
  - [The Ringer Algorithm](#the-ringer-algorithm)
  - [Specialist Committee Architecture](#specialist-committee-architecture)
  - [Quantization for Hardware Deployment](#quantization-for-hardware-deployment)
- [Setup & Environment](#setup--environment)
  - [uv Setup (Recommended)](#uv-setup-recommended)
  - [Docker Setup](#docker-setup)
- [CLI Workflows (`python cli.py ringer`)](#cli-workflows-python-clipy-ringer)
  - [1. Specialist Training (`training`)](#1-specialist-training-training)
  - [2. Decision Threshold Fitting (`threshold-fit`)](#2-decision-threshold-fitting-threshold-fit)
  - [3. Full-Dataset Inference (`inference`)](#3-full-dataset-inference-inference)
  - [4. Normalization Quantization Analysis (`alternative-norm1`)](#4-normalization-quantization-analysis-alternative-norm1)
- [Distributed Execution (`ExecutorConfig`)](#distributed-execution-executorconfig)
- [Repository Structure](#repository-structure)
- [Notebooks & Analysis](#notebooks--analysis)
- [Developer & Agent Guidelines](#developer--agent-guidelines)

---

## Overview & Physics Context

### The Ringer Algorithm
In high-energy particle physics, fast electron and photon identification in real-time trigger systems requires discriminative representations of calorimeter energy deposits. The **Ringer algorithm** concentrates shower development into concentric calorimeter rings around the most energetic cluster cell across the 7 longitudinal calorimeter sampling layers of the ATLAS detector:

| Calorimeter Sampling Layer | Number of Rings | Description |
|:---|:---:|:---|
| **Pre-Sampler (PS)** | 8 | Initial shower initiation |
| **Electromagnetic 1 (EM1)** | 64 | Fine strip granularity for $\gamma / \pi^0$ separation |
| **Electromagnetic 2 (EM2)** | 8 | Core electromagnetic shower containment |
| **Electromagnetic 3 (EM3)** | 8 | Tail of the EM shower |
| **Hadronic 1 (Had1)** | 4 | Initial hadronic leakage |
| **Hadronic 2 (Had2)** | 4 | Mid hadronic leakage |
| **Hadronic 3 (Had3)** | 4 | Deep hadronic leakage |
| **Total** | **100** | Full calorimeter shower profile |

Before feeding rings into neural networks, ring energies are normalized (using `AlternativeNorm1`, which normalizes ring deposits while preserving signed energy ratios) and optionally decimated by layer-wise fractions (`ring_fraction`).

### Specialist Committee Architecture
Electromagnetic shower shapes vary substantially with transverse energy ($E_T$) and pseudorapidity ($|\eta|$). To achieve optimal classification:
1. The $(E_T, |\eta|)$ phase space is divided into a 2D grid of specialist bins.
2. For each bin, a dedicated Multi-Layer Perceptron (MLP) specialist is trained using $k$-fold cross-validation and multiple random weight initializations.
3. The best model per bin is selected based on the **Signal-to-Background Performance index ($SP$)**:
   $$\text{SP} = \sqrt{\sqrt{\text{TPR} \cdot (1 - \text{FPR})} \cdot \frac{\text{TPR} + (1 - \text{FPR})}{2}}$$
4. The ensemble of selected models forms a **Binned Specialist Committee** (`BinnedSpecialistCommittee`), routing events at inference time to their corresponding specialist based on their $(E_T, |\eta|)$ coordinate.

### Quantization for Hardware Deployment
Trigger algorithms at Level 0 operate within stringent FPGA latency and resource constraints (e.g., target latencies $\le 1\,\mu\text{s}$). The framework supports:
- **Floating-Point (FP)** baseline training.
- **Post-Training Quantization (PTQ)**: Quantizing weights and normalizer constants to custom fixed-point representations $\langle \text{total\_bits}, \text{integer\_bits} \rangle$.
- **Quantization-Aware Training (QAT)**: Simulating fixed-point quantization during gradient descent using `hgq2` (Heterogeneous Graphical Quantization) or custom quantization layers.

---

## Setup & Environment

### uv Setup (Recommended)

The project manages its environment and dependencies using [uv](https://docs.astral.sh/uv/) and targets **Python 3.12+**.

1. **Install dependencies and sync the virtual environment**:
   ```bash
   uv sync
   ```
   To include development tools (`pytest`, `ruff`):
   ```bash
   uv sync --all-groups
   ```

2. **Set the Keras backend (Keras 3 with TensorFlow)**:
   ```bash
   export KERAS_BACKEND=tensorflow
   ```

`uv run` automatically executes commands within the project's virtual environment:
```bash
uv run python cli.py ringer --help
```

### Docker Setup

Alternatively, build and run the development container using Docker Compose:

```bash
docker compose build
docker compose up -d
docker compose exec boosted-lorenzetti bash
```

---

## CLI Workflows (`python cli.py ringer`)

The command-line interface is exposed via [cli.py](./cli.py) under the `ringer` command group:

```bash
python cli.py ringer --help
```

Available subcommands:
- `training`: Train specialist committee MLPs (FP / QAT) across folds and initializations.
- `threshold-fit`: Fit decision thresholds to target operating points (Tight, Medium, Loose) and package deployment committees.
- `inference`: Run high-throughput, memory-efficient streaming inference over full datasets.
- `alternative-norm1`: Quantization error analysis (MSE, MAE, KL divergence) for ring normalization across bit widths.

---

### 1. Specialist Training (`training`)

Trains specialist neural networks over configured $(E_T, |\eta|)$ bins, folds, and initializations.

```bash
python cli.py ringer training --config configs/training_job.yaml
```

#### Example Configuration (`configs/training_job.yaml`)

```yaml
# Dataset configuration
dataset_dir: /path/to/parquet_dataset_dir
data_table: electron_ringer
kfold_table: standard_binning_kfold
rings_col: TrigEMClusterContainer.ringsE
label_col: label
fold_col: kfold
et_col: TrigEMClusterContainer.et
eta_col: TrigEMClusterContainer.eta
norm_strategy: l1
ring_fraction: 2
batch_size: 1024

# Phase space binning
et_bins:
  - 15000.0
  - 20000.0
  - 30000.0
  - 40000.0
  - 50000.0
  - .inf
eta_bins:
  - 0.0
  - 0.8
  - 1.37
  - 1.54
  - 2.37
  - 2.5

# Model architecture & optimization
model_factory:
  object_type: mlp
  layers:
    - units: 16
      activation: relu
      name: dense_0
    - units: 1
      activation: sigmoid
      name: output_dense
  name: ringer_mlp

loss:
  object_type: binary_cross_entropy
  from_logits: false

optimizer:
  object_type: adam
  learning_rate: 0.001

epochs: 500
patience: 25
inits: 5
balance_class_weights: true

# Selection criteria for best initialization and fold
best_init:
  key: fit.val.max_sp
  criterion: max
best_fold:
  key: fit.val.max_sp
  criterion: max

output_path: ./output/ringer_training
executor_config:
  executor_type: debug  # debug, local, or slurm
  cpus_per_task: 1
  logs_dir: ./logs/training
  name: ringer_training
```

> [!TIP]
> To print the complete schema and all supported parameters for `training`, run:
> ```bash
> python cli.py ringer training --help
> ```

---

### 2. Decision Threshold Fitting (`threshold-fit`)

Evaluates a completed training job, locates the decision threshold for each specialist to match reference operating points (e.g. Tight, Medium, Loose True Positive Rates), and exports standalone deployment JSON configuration files.

```bash
python cli.py ringer threshold-fit --config configs/threshold_fit_job.yaml
```

#### Example Configuration (`configs/threshold_fit_job.yaml`)

```yaml
dataset_dir: /path/to/parquet_dataset_dir
data_table: electron_ringer
kfold_table: standard_binning_kfold
rings_col: TrigEMClusterContainer.ringsE
label_col: label
fold_col: kfold
et_col: TrigEMClusterContainer.et
eta_col: TrigEMClusterContainer.eta

# Reference operating points to fit
references:
  tight:
    tpr: 0.95
    color: red
    label: Tight
  medium:
    tpr: 0.97
    color: orange
    label: Medium
  loose:
    tpr: 0.99
    color: green
    label: Loose

training_output_path: ./output/ringer_training
output_path: ./output/ringer_threshold_fit

executor_config:
  executor_type: debug
  cpus_per_task: 1
  logs_dir: ./logs/threshold_fit
  name: threshold_fit
```

#### Output Structure
```
output_path/
├── config.json
├── models/
│   └── member_{id}/
│       ├── model.keras
│       └── test_confusion_matrix.parquet
├── tight.json   # Self-contained committee deployment configuration
├── medium.json
└── loose.json
```

---

### 3. Full-Dataset Inference (`inference`)

Runs full-dataset inference using a fitted committee. Results are streamed directly to a Parquet table using Polars streaming mode (`sink_parquet`), preventing out-of-memory errors on large datasets:

```bash
python cli.py ringer inference --config configs/inference_job.yaml
```

#### Example Configuration (`configs/inference_job.yaml`)

```yaml
dataset_dir: /path/to/parquet_dataset_dir
data_table: electron_ringer
kfold_table: standard_binning_kfold
rings_col: TrigEMClusterContainer.ringsE
label_col: label
fold_col: kfold
et_col: TrigEMClusterContainer.et
eta_col: TrigEMClusterContainer.eta

threshold_fit_output_path: ./output/ringer_threshold_fit
output_path: ./output/ringer_inference

executor_config:
  executor_type: debug
  cpus_per_task: 1
  logs_dir: ./logs/inference
  name: ringer_inference
```

#### Output Columns (`inference_results.parquet`)
- `id` (UInt64): Event identifier for joins.
- `{op}.output` (Float32): Raw continuous committee score for operating point `{op}`.
- `{op}.prediction` (Boolean): Discrete binary trigger decision (accepted / rejected).

---

### 4. Normalization Quantization Analysis (`alternative-norm1`)

Performs a grid sweep over integer and fractional bit widths to measure the quantization error of `AlternativeNorm1` against floating-point values:

```bash
python cli.py ringer alternative-norm1 --config configs/alt_norm1_job.yaml
```

Metrics evaluated include Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE), and Kullback–Leibler (KL) divergence.

---

## Distributed Execution (`ExecutorConfig`)

All jobs use [submitit](https://github.com/facebookincubator/submitit) via `ExecutorConfig` for unified execution across environments:

| `executor_type` | Use Case | Description |
|:---|:---|:---|
| `debug` | Local debugging & testing | Runs synchronously in the active process. |
| `local` | Multi-core workstation | Runs parallel subprocesses on the local machine. |
| `slurm` | HPC Cluster | Submits array jobs to a SLURM workload manager. |

#### Example SLURM Configuration
```yaml
executor_config:
  executor_type: slurm
  cpus_per_task: 10
  slurm_partition: cpu
  slurm_array_parallelism: 6
  timeout_min: 240
  stderr_to_stdout: true
  logs_dir: /path/to/cluster/logs
  name: ringer_job
```

---

## Repository Structure

```
neuralnet/
├── cli.py                     # Typer CLI entrypoint exposing `ringer` command group
├── Dockerfile                 # Container image specification
├── docker-compose.yml         # Container compose file
├── pyproject.toml             # Ruff, Pytest, and project dependencies
├── uv.lock                    # Resolved dependency lockfile
│
├── neuralnet/                 # Core Python package
│   ├── workflows/             # End-to-end execution workflows
│   │   └── ringer/            # Active Ringer workflows (training, threshold_fit, inference, alt_norm1)
│   ├── models/                # Neural network architectures & factories
│   │   ├── dense.py           # Keras MLP factories & fixed-point layers
│   │   ├── binned_committee.py# Generic binned specialist committee abstraction
│   │   ├── kan.py / vkan.py   # Kolmogorov–Arnold Networks (KAN) implementations
│   │   └── quantum.py         # Quantum neural network layers (Pennylane)
│   ├── quantization/          # Quantization tools (FixedPointQuantizer, HGQ2, Keras layers)
│   ├── normalizers/           # Ring normalization routines (AlternativeNorm1, Polars, factories)
│   ├── datasets/              # Dataset wrappers (RingerParquetDataset, ParquetDataset)
│   ├── metrics.py             # Performance metrics (SP index, confusion matrix, ROC/AUC)
│   ├── bins.py                # VariableBin and AbsoluteVariableBin definitions
│   ├── submitit.py            # Unified execution abstraction (debug, local, slurm)
│   ├── plotting.py            # Publication-grade plotting with Matplotlib & mplhep
│   └── pydantic.py            # YamlBaseModel & schema generation helpers
│
├── tests/                     # Unit and integration test suite
│   ├── test_workflows/        # Integration tests for ringer training, inference, alt_norm1
│   ├── test_models/           # Tests for dense, binned committee, quantum, and KAN models
│   ├── test_quantization/     # Tests for fixed-point quantizers and Keras quantization
│   └── test_normalizers.py    # Normalization tests
│
├── notebooks/                 # Exploratory data analysis, threshold fit, and evaluation notebooks
├── jobs/                      # Production configuration YAMLs and job artifacts
└── docs/                      # Documentation and UML diagrams
```

---

## Notebooks & Analysis

The [notebooks/](./notebooks/) directory provides reproducible analysis workflows:
- [explorative_analysis.ipynb](./notebooks/explorative_analysis.ipynb): Signal-to-background distributions and ring energy profiles.
- [ringer_committee_threshold_fit.ipynb](./notebooks/ringer_committee_threshold_fit.ipynb): Interactive inspection of operating point fits and ROC curves.
- [model_comparison.ipynb](./notebooks/model_comparison.ipynb): Performance comparison between FP, PTQ, and QAT specialist committees.
- [quadrant_plot.ipynb](./notebooks/quadrant_plot.ipynb): Quadrant response analysis across detector regions.

---

## Developer & Agent Guidelines

For developer conventions, coding standards, Polars / Pydantic design patterns, and instructions for AI coding assistants, see [AGENTS.md](./AGENTS.md).

