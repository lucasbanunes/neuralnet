# Tasks

## 1. On-Disk Migration Core (`src/neuralnet/workflows/ringer/migration.py`)

- [ ] 1.1 Create `src/neuralnet/workflows/ringer/migration.py` implementing `migrate_training_config(config: dict[str, Any]) -> dict[str, Any]` (translating `"fixed_point_quantized_mlp"` to `"mlp"` and layer `"floating_bits"` to `"fractional_bits"` without mutating the input dict, rejecting conflicting `"floating_bits"` + `"fractional_bits"`) and `migrate_keras_archive(src_path: Path, dst_path: Path) -> bool` (rewriting `"floating_bits"` to `"fractional_bits"` in `FixedPointQuantizationDense` entries inside the `.keras` ZIP's `config.json` while copying `metadata.json` and `model.weights.h5` byte-for-byte); verify with unit tests in `tests/test_workflows/test_ringer_migration.py` including loading a migrated `.keras` archive with `keras.models.load_model(..., custom_objects=get_custom_objects())` and asserting identical predictions (`uv run pytest tests/test_workflows/test_ringer_migration.py`)
- [ ] 1.2 Implement `migrate_training_job_dir(input_path: Path, output_path: Path, overwrite: bool = False)` in `src/neuralnet/workflows/ringer/migration.py` to validate the source directory, enforce `input != output` and `overwrite` safety checks, migrate `config.json` and all `models/member_*/model.keras` archives, copy `all_models_results.parquet`, `selected_models.parquet`, and `models/member_*/results.json.zip`, and validate the migrated directory via `RingerKerasTrainingJob.load(output_path)`; verify with unit tests in `tests/test_workflows/test_ringer_migration.py` covering success, source immutability, `input == output` rejection, and `overwrite=False` / `overwrite=True` behavior (`uv run pytest tests/test_workflows/test_ringer_migration.py`)

## 2. CLI Command Registration (`src/neuralnet/workflows/ringer/cli.py`)

- [ ] 2.1 Register `migrate` command (`--input` / `-i`, `--output` / `-o`, `--overwrite`) on `ringer_app` in `src/neuralnet/workflows/ringer/cli.py` with clear help text and error reporting; verify via Typer `CliRunner` tests in `tests/test_workflows/test_ringer_migration.py` covering successful invocation, `--help`, and non-zero exit codes on invalid paths or existing output without `--overwrite` (`uv run pytest tests/test_workflows/test_ringer_migration.py`)

## 3. End-to-End Integration & Suite Verification

- [ ] 3.1 Add an end-to-end integration test in `tests/test_workflows/test_ringer_migration.py` that migrates a temporary legacy custom fixed-point training job directory, runs `RingerCommitteeThresholdFitJob` (`executor_type="debug"`) against the migrated directory, and reloads the exported operating-point committee via `BinnedSpecialistCommittee.from_json()` for inference; verify with `uv run pytest tests/test_workflows/test_ringer_migration.py`
- [ ] 3.2 Run the full test suite and linter (`uv run pytest` and `uv run ruff check .`) and confirm zero regressions or lint/formatting issues
