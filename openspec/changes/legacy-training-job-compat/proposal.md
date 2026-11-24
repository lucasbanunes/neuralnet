# Proposal

## Why

After the core refactor (`reorganize-core-and-yaml-activities`), running `neuralnet ringer threshold-fit` against pre-refactor custom fixed-point training jobs (specifically `q715`) fails at both configuration loading and Keras model deserialization because persisted `config.json` files reference `fixed_point_quantized_mlp` and `floating_bits`, and saved `.keras` archives serialize `FixedPointQuantizationDense` with `floating_bits`. Because `q715` is a one-off legacy job to be migrated and committed in the new format, we prefer an explicit on-disk migration CLI command over maintaining runtime legacy compatibility shims in core loaders and Keras layers.

## What Changes

- Add an on-disk migration module (`src/neuralnet/workflows/ringer/migration.py`) and register a new CLI command `neuralnet ringer migrate --input <src> --output <dst>` (with optional `--overwrite`) under `src/neuralnet/workflows/ringer/cli.py`:
  - Rewrite `config.json` in the destination directory, mapping `model_factory.object_type: "fixed_point_quantized_mlp"` to `"mlp"` and layer field `"floating_bits"` to `"fractional_bits"`, then validate the migrated config against the strict current `RingerKerasTrainingJob` schema.
  - Rewrite each `models/member_*/model.keras` ZIP archive into the destination directory by replacing `"floating_bits"` with `"fractional_bits"` inside the archive's `config.json` while preserving `model.weights.h5` and `metadata.json` byte-for-byte.
  - Copy `all_models_results.parquet`, `selected_models.parquet`, and per-member `results.json.zip` into the destination directory while leaving the source directory completely unmodified.
  - Enforce safety checks: refuse `input == output` (preventing in-place mutation) and refuse to overwrite an existing output directory unless `--overwrite` is explicitly passed.
- Keep `RingerKerasTrainingJob.load()`, `from_yaml()`, and `FixedPointQuantizationDense` strictly on the post-refactor schema with zero runtime legacy shims.
- Add unit, CLI, and end-to-end integration tests verifying migration of legacy `config.json` and `.keras` archives, safety guardrails, and full `RingerCommitteeThresholdFitJob` + `BinnedSpecialistCommittee.from_json()` execution on a migrated directory.
- Out of scope: pre-refactor HGQ2 training jobs (`hgq2_mlp` / legacy `HGQFixedPointConfig`) and read-time runtime shims in `training.py` or `quantization/keras.py`.

## Capabilities

### New Capabilities
- `legacy-artifact-compatibility`: Explicit CLI-driven on-disk migration of persisted pre-refactor custom fixed-point training job directories (`config.json`, `.keras` member models, and results tables) into a new directory conforming to the modern post-refactor schema.

### Modified Capabilities
<!-- No existing capability requirements are changing; core loaders, Keras layers, and YAML schemas remain strictly validated under pluggable-quantization and model-layer-factories. -->

## Impact

- **New module**: `src/neuralnet/workflows/ringer/migration.py`
- **Modified module**: `src/neuralnet/workflows/ringer/cli.py` (register `migrate` command on `ringer_app`)
- **Unchanged core modules**: `src/neuralnet/workflows/ringer/training.py` and `src/neuralnet/quantization/keras.py` remain free of legacy runtime shims.
- **Tests**: New unit, CLI, and workflow integration tests in `tests/test_workflows/test_ringer_migration.py`.
- **Downstream workflows**: Once migrated via `uv run neuralnet ringer migrate --input <old> --output <new>`, the new directory works natively with `RingerKerasTrainingJob.load()`, `RingerCommitteeThresholdFitJob`, `RingerCommitteeInferenceJob`, and `BinnedSpecialistCommittee.from_json()`, while already-compatible floating-point jobs (such as `run3_mlp`) require no migration.
