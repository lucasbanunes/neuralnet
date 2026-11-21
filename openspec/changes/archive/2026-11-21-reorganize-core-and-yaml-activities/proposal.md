# Proposal

## Why

The `neuralnet` repository has grown organically around YAML-driven Ringer workflows, but its core building blocks have fragmented: fixed-point quantization is split across three inconsistent representations (`FixedPointQuantizer`, custom `FixedPointQuantizationDense`, and `HGQFixedPointConfig` which is not even wired into training and cannot express learnable bitwidth ranges), layer and model factories are split across `models/dense.py`, `models/kan.py`, `models/quantum.py`, `layers/quantum.py`, and `models/keras/factories.py`, and `BinnedSpecialistCommittee` is duplicated in two modules. Reorganizing the core library around unified quantization nomenclature, explicit per-layer factories, unified model factories, and a shared `BaseActivity` contract establishes a clean, reproducible YAML-activity foundation before adding future quantization techniques or migrating notebook analyses to YAML.

## What Changes

- **Pluggable Quantization Architecture (`neuralnet.quantization`)**:
  - Introduce a discriminated-union (`object_type`) quantization configuration and strategy protocol supporting both custom fixed-point quantization and `hgq2`, designed for drop-in future quantization techniques not available in `hgq2`.
  - Standardize bitwidth field names exclusively to `integer_bits` and `fractional_bits` across custom fixed-point quantization (`FixedPointQuantizer`, `FixedPointQuantizationDense`, `FixedPointQuantizedDenseFactory`, and Polars normalizers) with **no backward-compatible YAML aliases** for `floating_bits`; migrate all existing repository YAMLs (`tests/data/*.yaml`, `jobs/**/*.yaml`) to the unified schema. **BREAKING** for YAML configs using `floating_bits`.
  - Keep native `hgq2` framework field names (`k0`, `i0`, `f0`, `ic`, `fc`) in `HGQ2QuantizerConfig`, extending `ic`/`fc` to support both constant constraints (`Constant`) and learnable bitwidth ranges (`MinMax`), enabling `HGQ2DenseFactory` (`QDense`) to optimize bitwidths during training.
  - Reuse `FixedPointQuantizer` in `normalizers/factories.py` (`FixedPointAlternativeNormL1`) instead of duplicating fields.
- **Unified Layer & Model Factories (`neuralnet.layers`, `neuralnet.models`)**:
  - Consolidate all Keras 3 layer factories into `neuralnet.layers` under a single `LayerType` discriminated union with an explicit factory per layer type: `DenseFactory` (strictly floating-point `Dense`, with no quantizer field), `FixedPointQuantizedDenseFactory` (using its own custom fixed-point quantizer logic), `HGQ2DenseFactory` (using its own HGQ2 range-aware quantizer logic), `KAN1DLayerFactory`, and PennyLane `Quantum` layer factories.
  - Update `KerasSequentialModelFactory` to support every layer in `LayerType` (`Dense`, custom fixed-point quantized `Dense`, `HGQ2` `Dense`, `KAN 1D`, `Quantum`) and validate `KERAS_BACKEND` compatibility when Torch-wrapped layers (KAN, Quantum) are used.
  - Provide specialized convenience model factories (`MLPFactory` accepting dense-family layers `DenseFactory | FixedPointQuantizedDenseFactory | HGQ2DenseFactory`, and `KANFactory`) alongside `KerasSequentialModelFactory` in a unified `ModelFactoryType` union wired into `RingerKerasTrainingJob`.
- **Duplicate Cleanup & Deprecation**:
  - Consolidate `neuralnet/models/binned_committee.py` and `neuralnet/workflows/ringer/models.py` into a single `BinnedSpecialistModel` / `BinnedSpecialistCommittee` implementation supporting pluggable quantization strategies. **BREAKING** for imports from the removed duplicate module path.
  - Retire the duplicate `neuralnet/models/quantum.py` in favor of `neuralnet/layers/quantum.py`. **BREAKING** for direct imports from `neuralnet.models.quantum`.
  - Mark `neuralnet/models/vkan.py` as deprecated (emitting a `DeprecationWarning` on import/use) while leaving `neuralnet/torch/`, `neuralnet/tensorflow/`, and `notebooks/` untouched.
- **Shared YAML Activity Foundation (`neuralnet.workflows`)**:
  - Introduce a shared `BaseActivity(YamlBaseModel)` base class defining common reproducibility and execution conventions (`output_path`, `logger_name`, config snapshot dumping, `submit()`/`run()`, and uniform Typer CLI registration with auto-generated schema help).
  - Refactor the existing 4 Ringer CLI activities (`training`, `threshold-fit`, `inference`, `alternative-norm1`) onto `BaseActivity` without adding new CLI activities in this change.

## Capabilities

### New Capabilities
- `pluggable-quantization`: Discriminated-union quantization configuration and extensible Keras/Polars strategy contracts supporting custom fixed-point, range-aware HGQ2, and future quantization techniques with unified `integer_bits`/`fractional_bits` nomenclature.
- `model-layer-factories`: Explicit Keras 3 layer factories (`DenseFactory`, `FixedPointQuantizedDenseFactory`, `HGQ2DenseFactory`, `KAN1DLayerFactory`, `Quantum`) composed via `KerasSequentialModelFactory` and specialized convenience model factories (`MLPFactory`, `KANFactory`), with backend compatibility validation.
- `yaml-activity-framework`: Shared `BaseActivity(YamlBaseModel)` contract, uniform CLI registration, and consolidated `BinnedSpecialistCommittee` backing the existing 4 Ringer YAML activities (`training`, `threshold-fit`, `inference`, `alternative-norm1`).

### Modified Capabilities
<!-- No existing specs in openspec/specs/ -->

## Impact

- **Affected Packages**:
  - `src/neuralnet/quantization/` (`quantizers.py`, `keras.py`, `hgq.py`, new `base.py`)
  - `src/neuralnet/layers/` (`quantum.py`, new `dense.py`, `kan.py`, `__init__.py`)
  - `src/neuralnet/models/` (`dense.py`, `kan.py`, `quantum.py`, `vkan.py`, `binned_committee.py`, `keras/factories.py`, new `sequential.py`, `mlp.py`, `committee.py`)
  - `src/neuralnet/normalizers/` (`factories.py`, `polars.py`)
  - `src/neuralnet/workflows/ringer/` (`cli.py`, `training.py`, `threshold_fit.py`, `inference.py`, `alternative_norm1.py`, `models.py`)
  - `src/neuralnet/workflows/base.py`
- **Affected YAML Configs & Tests**:
  - `tests/data/*.yaml`, `jobs/**/*.yaml` (migrated to unified `integer_bits`/`fractional_bits` and explicit layer factory schemas)
  - `tests/test_models/`, `tests/test_quantization/`, `tests/test_normalizers.py`, `tests/test_workflows/`
- **Untouched / Out of Scope**:
  - `notebooks/`, `src/neuralnet/torch/`, `src/neuralnet/tensorflow/`, and adding new CLI activities (e.g., PTQ or comparison workflows) are explicitly out of scope for this change.
