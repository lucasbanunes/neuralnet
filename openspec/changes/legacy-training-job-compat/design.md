# Design

## Context

See `proposal.md` for motivation. After commit `6ce2f9e`, pre-refactor custom fixed-point training directories (specifically `q715`) fail at two points when consumed by downstream workflows:

1. **Persisted `config.json`** (`RingerKerasTrainingJob.load()`):
   - `model_factory.object_type` is `"fixed_point_quantized_mlp"` instead of `"mlp"`.
   - `model_factory.layers[*]` uses `"floating_bits"` instead of `"fractional_bits"`.
2. **Persisted `models/member_*/model.keras`** (`keras.models.load_model(..., custom_objects=get_custom_objects())`):
   - The `.keras` archive's internal `config.json` serializes `neuralnet>FixedPointQuantizationDense` with `"floating_bits"` instead of `"fractional_bits"`.
   - Empirical inspection of Keras 3 `.keras` archives confirms `model.weights.h5` stores layer variables by positional index (`vars/0..3`: `kernel`, `bias`, `fractional_bits`/`floating_bits`, `integer_bits`), not by variable name.

Pre-refactor floating-point MLP directories (such as `run3_mlp`) already use `object_type: "mlp"`, `layers[*].object_type: "dense"`, and standard `keras.layers.Dense`, so they require no migration.

## Goals / Non-Goals

**Goals:**
- Provide an explicit, non-destructive CLI command (`neuralnet ringer migrate --input <src> --output <dst>`) that produces a self-contained, modern training job directory ready to be committed and consumed by `threshold-fit`, `inference`, and notebooks.
- Keep `RingerKerasTrainingJob.load()`, `from_yaml()`, and `FixedPointQuantizationDense` 100% strict with zero legacy runtime shims.
- Migrate all member `.keras` archives quickly and deterministically without altering trained weights.

**Non-Goals:**
- Supporting pre-refactor HGQ2 training jobs (`hgq2_mlp` / old `HGQFixedPointConfig`).
- Adding runtime compatibility shims to `RingerKerasTrainingJob.load()` or `FixedPointQuantizationDense.from_config()`.
- Performing in-place mutation of the source training job directory.

## Decisions

### 1. Explicit on-disk migration CLI vs. read-time runtime shims
- **Choice**: Explicit on-disk migration via `neuralnet ringer migrate --input <src> --output <dst>`.
- **Alternatives considered**:
  - *Read-time compatibility shims in `RingerKerasTrainingJob.load()` and `FixedPointQuantizationDense.from_config()`*: Avoids duplicating the directory on disk, but keeps legacy shim code inside core runtime modules indefinitely even though only `q715` needs migration.
- **Rationale**: Since `q715` is the only legacy custom fixed-point job and the user wants to commit the migrated data structure, an explicit migration command produces a clean native artifact while keeping core training and quantization modules completely free of legacy branches.

### 2. ZIP-level `.keras` JSON rewrite vs. full Keras load-and-resave
- **Choice**: Migrate each `model.keras` archive at the ZIP/JSON level: read `config.json` from the source `.keras` ZIP, recursively rewrite any `FixedPointQuantizationDense` layer config `"floating_bits"` → `"fractional_bits"` (raising `ValueError` if both are present), and write the updated `config.json` alongside byte-for-byte copies of `metadata.json` and `model.weights.h5` into the destination `.keras` ZIP.
- **Alternatives considered**:
  - *Load with a temporary Keras custom object and call `model.save()`*: Requires TensorFlow/Keras initialization and graph/weight rebuild for every member (125+ models), is much slower, and still requires a Keras deserialization shim.
- **Rationale**: Because Keras 3 stores weights in `model.weights.h5` by positional index (`vars/0..3`), rewriting only the archive's `config.json` is fast, deterministic, guarantees bit-identical weights, and needs no Keras runtime shim at all.

### 3. Dedicated migration module and CLI registration
- **Choice**: Implement migration logic in `src/neuralnet/workflows/ringer/migration.py` with three focused functions:
  - `migrate_training_config(config: dict[str, Any]) -> dict[str, Any]`
  - `migrate_keras_archive(src_path: Path, dst_path: Path) -> bool` (returns whether a legacy rewrite was applied)
  - `migrate_training_job_dir(input_path: Path, output_path: Path, overwrite: bool = False) -> MigrationSummary`
- Register `migrate` directly as a Typer command on `ringer_app` in `src/neuralnet/workflows/ringer/cli.py` with `--input` / `-i`, `--output` / `-o`, and `--overwrite` (default `False`).
- **Rationale**: `migrate` is a one-shot filesystem utility rather than a reproducible YAML-configured experiment activity (`BaseActivity`), so direct `--input`/`--output` CLI options are more ergonomic than requiring a YAML file.

### 4. Safety guardrails and post-migration validation
- Validate source directory structure upfront using `RingerKerasTrainingJob.validate_saved_directory(input_path)`.
- Reject `input_path.resolve() == output_path.resolve()` with a clear `ValueError` / non-zero CLI exit.
- Reject an existing `output_path` unless `overwrite=True` (when `overwrite=True`, remove or replace `output_path` cleanly only after validating `input_path` and `input_path != output_path`).
- After writing the migrated `config.json`, validate it by calling `RingerKerasTrainingJob(**{**migrated_config, "output_path": output_path})` (and `RingerKerasTrainingJob.load(output_path)` after all files are written) so any schema mismatch fails immediately.
- Note on `output_path` in `config.json`: update `output_path` in the migrated `config.json` to `str(output_path)` if present (matching `BaseActivity.dump_config()`), while `RingerKerasTrainingJob.load()` continues to bind `output_path` dynamically on load.

## Risks / Trade-offs

- **Keras 3 positional weight loading assumption** → Verified empirically that `FixedPointQuantizationDense` saves variables in `model.weights.h5` as positional `vars/0..3`. Mitigated with an automated test that migrates a synthesized legacy `.keras` archive, loads it with `keras.models.load_model(..., custom_objects=get_custom_objects())`, and asserts bit-identical predictions.
- **Additional files in source directory (e.g., `logs/`, `config.yaml`)** → Copy the canonical training job artifacts (`config.json`, `all_models_results.parquet`, `selected_models.parquet`, and `models/member_*/{model.keras, results.json.zip}`). Other non-canonical files (such as SLURM `logs/`) are not required by `RingerKerasTrainingJob` and do not need to be duplicated, keeping the migrated directory clean for committing.
