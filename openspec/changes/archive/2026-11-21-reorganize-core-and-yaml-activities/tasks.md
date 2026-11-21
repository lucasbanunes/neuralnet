# Tasks

## 1. Pluggable Quantization Architecture

- [x] 1.1 Create `src/neuralnet/quantization/base.py` defining `KerasQuantizerProtocol`, `PolarsQuantizerProtocol`, the `QuantizerConfig` discriminated union (`object_type`), and `get_custom_objects()`, and verify unit tests confirm protocol compliance and custom-object discovery.
- [x] 1.2 Unify custom fixed-point quantization (`src/neuralnet/quantization/quantizers.py` and `src/neuralnet/quantization/keras.py`) around `CustomFixedPointQuantizer` (`object_type: "custom_fixed_point"`) using `integer_bits` and `fractional_bits` exclusively (removing `floating_bits` with no retrocompatible aliases), and verify `uv run pytest tests/test_quantization/` passes.
- [x] 1.3 Refactor `src/neuralnet/quantization/hgq.py` to expose `HGQ2QuantizerConfig` (`object_type: "hgq2_fixed_point"`) keeping native `hgq2` field names (`k0`, `i0`, `f0`, `ic`, `fc`) where `ic`/`fc` support `Constant` (defaulting to `i0`/`f0`) and `MinMax(min_value, max_value)` bitwidth range constraints, implementing `as_hgq_quantizer_config`, `quantize_model`, and `custom_objects`, and verify unit tests cover fixed bitwidths, learnable `MinMax` ranges, invalid range rejection, PTQ weight copy, and `.keras` save/reload round-trip.
- [x] 1.4 Update `src/neuralnet/normalizers/factories.py` (`FixedPointAlternativeNormL1`) to reuse `FixedPointQuantizer` bitwidth validation and verify `uv run pytest tests/test_normalizers.py` passes.

## 2. Unified Keras 3 Layer Factories

- [x] 2.1 Create `src/neuralnet/layers/dense.py` with explicit per-layer factories — `DenseFactory` (`"dense"`, strictly floating-point with no quantizer field), `FixedPointQuantizedDenseFactory` (`"fixed_point_quantized_dense"`, using its own `CustomFixedPointQuantizer` logic), and `HGQ2DenseFactory` (`"hgq2_dense"`, using its own native range-aware `HGQ2QuantizerConfig` logic) — plus `DenseLayerType` discriminated union, and verify unit tests cover each factory, `extra="forbid"` rejection of quantization fields on `DenseFactory`, rejection of legacy `floating_bits` on custom fixed-point layers, and `HGQ2DenseFactory` `MinMax` bit optimization config.
- [x] 2.2 Move `KAN1DLayerFactory` into `src/neuralnet/layers/kan.py` with explicit `KERAS_BACKEND=torch` validation and actionable error messaging, and verify unit tests cover both invalid-backend `ValueError` and valid schema parsing.
- [x] 2.3 Add consistent `KERAS_BACKEND=torch` validation to `src/neuralnet/layers/quantum.py`, remove the duplicate `src/neuralnet/models/quantum.py`, update imports, and verify `uv run pytest tests/test_models/test_quantum.py` passes.
- [x] 2.4 Export the unified `LayerType` discriminated union (`DenseFactory`, `FixedPointQuantizedDenseFactory`, `HGQ2DenseFactory`, `KAN1DLayerFactory`, and PennyLane quantum layer factories) from `src/neuralnet/layers/__init__.py` and verify unit tests parse mixed-layer lists from YAML/dicts and reject unknown `object_type` values.

## 3. Unified Model Factories, Committee Consolidation & VKAN Deprecation

- [x] 3.1 Create `src/neuralnet/models/sequential.py` (`KerasSequentialModelFactory` accepting `list[LayerType]`), `src/neuralnet/models/mlp.py` (`MLPFactory` accepting `list[DenseLayerType]`), and `src/neuralnet/models/kan.py` (`KANFactory` compiling to `keras.Sequential`), and export `ModelFactoryType` from `src/neuralnet/models/__init__.py`; verify unit tests in `tests/test_models/test_dense.py` and `tests/test_models/test_vkan.py` (or new `test_kan.py`) pass.
- [x] 3.2 Consolidate `src/neuralnet/models/binned_committee.py` and `src/neuralnet/workflows/ringer/models.py` into a single canonical `src/neuralnet/models/committee.py` (`BinnedSpecialistModel` and `BinnedSpecialistCommittee` supporting pluggable `QuantizerConfig` and `get_custom_objects()`), remove `binned_committee.py`, update imports, and verify `uv run pytest tests/test_models/test_binned_committee.py` passes.
- [x] 3.3 Deprecate `src/neuralnet/models/vkan.py` with a `DeprecationWarning` on import pointing to `KANFactory` and Ringer YAML workflows, and verify a unit test asserts the warning is emitted while leaving `src/neuralnet/torch/`, `src/neuralnet/tensorflow/`, and `notebooks/` untouched.

## 4. Shared YAML Activity Foundation, Ringer Workflows & YAML Migration

- [x] 4.1 Create `src/neuralnet/workflows/base.py` defining `BaseActivity(YamlBaseModel, ABC)` (`extra="forbid"`, `frozen=True`, `output_path`, `logger_name`, `dump_config()`, abstract `submit()`) and `register_activity_command(...)` for uniform Typer CLI registration with auto-generated markdown schema help, and verify unit tests cover `from_yaml`, `dump_config`, and unknown-key rejection.
- [x] 4.2 Refactor `RingerKerasTrainingJob` (`src/neuralnet/workflows/ringer/training.py`) to inherit from `BaseActivity` and accept `ModelFactoryType` (enabling `keras_sequential`, `mlp` with explicit `dense`/`fixed_point_quantized_dense`/`hgq2_dense` layers, and `kan`), migrate all existing repository YAMLs (`tests/data/*.yaml`, `jobs/ringer_mlp0_fp/config.yaml`, `jobs/ringer_mlp0_qat/config.yaml`, `jobs/basic_entangler/basic_entangler.yaml`, `jobs/hardware_efficient.yaml`, `jobs/strongly_entangling.yaml`) to the unified `integer_bits`/`fractional_bits` and explicit layer factory schema (no retrocompatible aliases), and verify `uv run pytest tests/test_workflows/test_ringer.py tests/test_workflows/test_ringer_quantization.py` pass.
- [x] 4.3 Refactor `RingerCommitteeThresholdFitJob`, `RingerCommitteeInferenceJob`, and `AlternativeNorm1Analysis` to inherit from `BaseActivity` and update `src/neuralnet/workflows/ringer/cli.py` to use `register_activity_command`, and verify `uv run pytest tests/test_workflows/test_alternative_norm1.py` and `uv run neuralnet ringer <cmd> --help` succeed for all 4 commands.

## 5. End-to-End Verification & Quality Checks

- [x] 5.1 Run the full test suite (`uv run pytest`) and confirm all tests pass across models, quantization, normalizers, and workflows.
- [x] 5.2 Run linter and formatter checks (`uv run ruff check .` and `uv run ruff format --check .`) and confirm zero issues.
- [x] 5.3 Update `AGENTS.md` module architecture notes if module paths changed and confirm all documented paths match the reorganized `src/neuralnet/` tree.
