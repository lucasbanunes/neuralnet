# AGENTS.md

> Developer and AI Agent Guidelines for the `neuralnet` repository.

This document serves as the technical guide for AI coding assistants and developers working on this codebase. It documents the domain context, architectural patterns, coding standards, common workflows, and pitfalls.

---

## 1. Domain Context & Physics Background

The `neuralnet` repository develops machine learning models and quantization pipelines for the **Neural Ringer Algorithm** deployed in the trigger system of the **ATLAS Experiment (CERN)**.

### Physics Overview
- **Trigger System**: ATLAS collides protons at $13.6\,\text{TeV}$ with a collision rate up to $40\,\text{MHz}$. Real-time filtering consists of hardware-level (Level 0 / Level 1) and software-level (Level 2 / Event Filter) triggers.
- **Ringer Concept**: The energy shower produced by electrons and photons in the ATLAS calorimeter exhibits cylindrical symmetry around the cluster centroid. The calorimeter cells are mapped to concentric rings across 7 longitudinal detector layers (Pre-Sampler, EM1, EM2, EM3, Had1, Had2, Had3), producing a 100-dimensional ring energy vector:
  - Pre-Sampler (PS): 8 rings
  - EM1: 64 rings
  - EM2: 8 rings
  - EM3: 8 rings
  - Had1: 4 rings
  - Had2: 4 rings
  - Had3: 4 rings
- **Ring Decimation (`ring_fraction`)**: To reduce input dimensionality and latency, rings can be decimated per layer (e.g., `ring_fraction: 2` takes every 2nd ring).
- **Energy Normalization (`AlternativeNorm1`)**: Raw ring energies are normalized while maintaining signed energy representation and stability against noise.
- **Phase Space Binned Specialists**: Shower profiles vary strongly with transverse energy ($E_T$) and pseudorapidity ($|\eta|$). The trigger architecture uses a **Binned Specialist Committee** (`BinnedSpecialistCommittee`):
  - Events are partitioned into $(E_T, |\eta|)$ intervals.
  - A dedicated Multi-Layer Perceptron (MLP) specialist is trained for each bin.
  - At inference time, each event is dynamically routed to its specialist model.
- **Key Metric: $SP$ Index**: The Signal-to-Background Performance index optimizes True Positive Rate (TPR) and False Positive Rate (FPR) simultaneously:
  $$SP = \sqrt{\sqrt{\text{TPR} \cdot (1 - \text{FPR})} \cdot \frac{\text{TPR} + (1 - \text{FPR})}{2}}$$
- **Quantization for Hardware (FPGA)**: Level 0 triggers are implemented on FPGAs with strict sub-microsecond latency and limited DSP/BRAM resources. Models must be quantized to fixed-point arithmetic $\langle \text{total\_bits}, \text{integer\_bits} \rangle$ via Post-Training Quantization (PTQ) or Quantization-Aware Training (QAT).

---

## 2. System Architecture & Key Modules

The repository follows a modular architecture separating datasets, models, quantization, workflows, and execution:

```
neuralnet/
├── cli.py                               # Typer CLI entrypoint exposing `ringer` command group
│
├── neuralnet/
│   ├── workflows/
│   │   └── ringer/
│   │       ├── cli.py                   # Typer subcommands for ringer workflows
│   │       ├── training.py              # RingerKerasTrainingJob (FP & QAT training)
│   │       ├── threshold_fit.py         # RingerCommitteeThresholdFitJob (operating point fitting)
│   │       ├── inference.py             # RingerCommitteeInferenceJob (streaming parquet inference)
│   │       ├── alternative_norm1.py     # AlternativeNorm1Analysis (quantization analysis)
│   │       ├── models.py                # BinnedSpecialistModel & BinnedSpecialistCommittee
│   │       └── fields.py                # Pydantic field validators & aliases
│   │
│   ├── models/
│   │   ├── dense.py                     # DenseFactory, MLPFactory, FixedPointQuantizedMLP
│   │   ├── binned_committee.py          # Generic binned specialist container
│   │   ├── kan.py / vkan.py             # Kolmogorov–Arnold Networks
│   │   └── quantum.py                   # Pennylane quantum neural layers
│   │
│   ├── quantization/
│   │   ├── quantizers.py                # FixedPointQuantizer (bitwidth & fractional math)
│   │   ├── keras.py                     # FixedPointQuantizationDense Keras layers
│   │   └── hgq.py                       # Heterogeneous Graphical Quantization (HGQ2) configs
│   │
│   ├── normalizers/
│   │   ├── polars.py                    # AlternativeNorm1 & FixedPointQuantizedAlternativeNorm1
│   │   └── factories.py                 # Pydantic factories for normalizers
│   │
│   ├── datasets/
│   │   ├── ringer.py                    # RingerParquetDataset & fold split management
│   │   └── __init__.py                  # Base ParquetDataset wrapper
│   │
│   ├── bins.py                          # VariableBin & AbsoluteVariableBin expressions
│   ├── metrics.py                       # sp_index, enhanced_confusion_matrix, Polars metrics
│   ├── submitit.py                      # ExecutorConfig (debug, local, slurm backend router)
│   ├── plotting.py                      # Matplotlib & mplhep publication plots
│   └── pydantic.py                      # YamlBaseModel & pydantic_to_markdown_schema
```

---

## 3. Core Design Principles & Coding Standards

When modifying or extending this repository, agents must adhere to the following principles:

### 3.1. Polars for Tabular Data
- **Never use Pandas** for core data manipulation; use **Polars** (`polars as pl`).
- Prefer **expressions** (`pl.col(...)`, `pl.when(...)`) over Python loops.
- Use **lazy execution** (`pl.LazyFrame`) whenever processing large datasets.
- For saving large inference tables, use **streaming execution** (`lazy_df.sink_parquet(...)`) to keep RAM usage minimal.
- Event tables are keyed by `id` (typically `pl.UInt64`), allowing relational joins between dataset tables, k-fold tables, and inference results.

### 3.2. Pydantic v2 & `YamlBaseModel`
- All configurations, jobs, and serializable factories must inherit from `YamlBaseModel` (or `BaseModel`).
- Set `model_config = ConfigDict(extra="forbid", frozen=True)` on job configuration classes to prevent invalid or accidental arguments.
- Document every field with `Annotated[T, Field(description="...")]`. These descriptions are automatically parsed by `pydantic_to_markdown_schema()` into CLI `--help` messages.
- For `pathlib.Path` fields that will be dumped to JSON, always attach `PlainSerializer(str, return_type=str)` to guarantee clean JSON serialization.

### 3.3. Keras 3 & Backend Hygiene
- The repository uses **Keras 3**. The primary backend for ringer training is **TensorFlow**.
- Always ensure `KERAS_BACKEND=tensorflow` is set before importing Keras modules if running outside standard CLI invocations:
  ```python
  import os
  os.environ.setdefault("KERAS_BACKEND", "tensorflow")
  import keras
  ```
- Use `from typing import TYPE_CHECKING` guards when importing heavyweight framework components (`keras`, `torch`, `hgq`) in modules that only need type annotations.

### 3.4. Execution via `ExecutorConfig` (Submitit)
- All long-running workflows (training, threshold-fitting, inference) must integrate with `ExecutorConfig` from `neuralnet.submitit`.
- Support three modes seamlessly:
  - `executor_type="debug"`: Executes in-process for immediate local debugging and automated tests.
  - `executor_type="local"`: Executes in parallel on multi-core local machines using Submitit `LocalExecutor`.
  - `executor_type="slurm"`: Submits jobs as an array job on SLURM clusters.

### 3.5. Formatting & Linting
- **Python Version**: Target Python 3.12+. Use modern Python 3.12 syntax (e.g. `type Alias = ...`, union operator `A | B`).
- **Formatter & Linter**: Configured via `pyproject.toml` with **Ruff**:
  - Line length: 120 characters.
  - Quote style: double quotes.
  - Indent style: spaces.
  - Run lint checks with: `uv run ruff check .`
  - Run formatting with: `uv run ruff format .`

### 3.6. Dependency & Environment Management (uv)
- Dependencies and virtual environments are managed natively using **`uv`**:
  - Install dependencies and sync environment: `uv sync`
  - Install development tools (`pytest`, `ruff`): `uv sync --all-groups`
  - Run commands in environment without activating: `uv run <command>`
  - Lockfile: `uv.lock` is tracked in version control; keep it updated with `uv lock`.

---

## 4. Key Workflows & CLI

The CLI is structured under `uv run python cli.py ringer <command> --config <path>`:

| Command | Workflow Class | Purpose | Primary Output |
|:---|:---|:---|:---|
| `training` | `RingerKerasTrainingJob` | Train binned MLPs across folds & initializations | `history.parquet`, `models/`, `results.parquet` |
| `threshold-fit` | `RingerCommitteeThresholdFitJob` | Fit operating point thresholds (Tight/Medium/Loose) | Standalone committee JSONs (`tight.json`, etc.) |
| `inference` | `RingerCommitteeInferenceJob` | Streaming full-dataset inference | `inference_results.parquet` (`id`, scores, flags) |
| `alternative-norm1` | `AlternativeNorm1Analysis` | Sweep bitwidths on ring energy normalization | Quantization error metrics table (`mse`, `kl_div`) |

---

## 5. Testing & Verification

### Running Tests
Tests are located in `tests/` and run via `uv run pytest`:

```bash
# Run all tests
uv run pytest

# Run ringer workflow integration tests
uv run pytest tests/test_workflows/test_ringer.py

# Run quantization tests
uv run pytest tests/test_workflows/test_ringer_quantization.py
uv run pytest tests/test_quantization/

# Run normalization tests
uv run pytest tests/test_workflows/test_alternative_norm1.py
```

### Pytest Conventions
- Tests use `tmp_path` fixtures for filesystem outputs.
- Test jobs use `executor_type="debug"` to execute synchronously within pytest.
- Ensure test dependencies are not written to the repository tree; always output to temporary fixtures or designated scratch directories.

---

## 6. Common Pitfalls & Anti-patterns

1. **Avoid In-Memory DataFrame Collects**: Do not call `.collect()` on large Polars DataFrames during inference or preprocessing. Use lazy scanning (`pl.scan_parquet`) and streaming sinks (`sink_parquet`).
2. **Avoid Hardcoded File Paths**: Never hardcode dataset or model paths in library modules. Use Pydantic configuration schemas and pass paths dynamically.
3. **Handle Infinity in YAML**: $E_T$ and $|\eta|$ bins often use infinity for open upper bounds. In YAML, write `.inf` or `inf` (which parses as `float("inf")`).
4. **Preserve Committee JSON Independence**: The operating-point JSON files generated by `threshold-fit` (e.g. `tight.json`) must be fully self-contained with relative paths to member models, allowing them to be loaded on inference clusters without needing the original training directory.
5. **Preserve Comments & Docstrings**: When editing existing code, preserve existing docstrings, math explanations, and comments.

