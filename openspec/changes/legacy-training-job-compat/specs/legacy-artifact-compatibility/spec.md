# Spec Delta

## Purpose

Provides an explicit CLI-driven on-disk migration utility that converts persisted pre-refactor custom fixed-point training job directories (`config.json`, `.keras` member models, and results artifacts) into a new directory conforming to the modern post-refactor schema without modifying the source directory or adding runtime legacy shims.

## ADDED Requirements

### Requirement: Migrate legacy training job configuration on disk
The system SHALL migrate a persisted pre-refactor custom fixed-point training job `config.json` into the modern post-refactor schema and validate the migrated configuration before writing it to the destination directory.

#### Scenario: Migrate a legacy fixed-point quantized MLP config
- **WHEN** a source training job directory contains a `config.json` with `model_factory.object_type: "fixed_point_quantized_mlp"` and layer entries specifying `"floating_bits"`
- **THEN** the destination `config.json` specifies `model_factory.object_type: "mlp"` and `"fractional_bits"` on each converted layer, passes strict `RingerKerasTrainingJob` schema validation, and leaves the source `config.json` untouched

#### Scenario: Reject ambiguous layer config containing both floating_bits and fractional_bits
- **WHEN** a source `config.json` contains a layer entry specifying both `"floating_bits"` and `"fractional_bits"`
- **THEN** migration fails with a clear error rather than silently overwriting either value

#### Scenario: Modern config passes through cleanly
- **WHEN** a source training job directory already uses the modern schema (e.g., `"mlp"` with `"dense"` or `"fixed_point_quantized_dense"` using `"fractional_bits"`)
- **THEN** migration writes a valid modern `config.json` to the destination directory without altering schema fields

### Requirement: Migrate legacy fixed-point Keras model archives on disk
The system SHALL migrate persisted pre-refactor custom fixed-point `.keras` member model archives so that serialized `FixedPointQuantizationDense` layer configs use `"fractional_bits"` instead of `"floating_bits"` while preserving trained weights verbatim.

#### Scenario: Migrate a legacy custom fixed-point .keras archive
- **WHEN** a source member directory contains a `model.keras` archive whose internal `config.json` serializes a `FixedPointQuantizationDense` layer with `"floating_bits"`
- **THEN** the destination `model.keras` archive serializes that layer with `"fractional_bits"` (and no `"floating_bits"` key), preserves the saved weights and metadata, loads cleanly through the standard custom-objects loader with no runtime compatibility shims, and produces identical predictions

#### Scenario: Reject ambiguous Keras layer config containing both floating_bits and fractional_bits
- **WHEN** a source `model.keras` archive contains a `FixedPointQuantizationDense` layer config with both `"floating_bits"` and `"fractional_bits"`
- **THEN** migration fails with a clear error

### Requirement: Safe non-destructive directory migration via CLI
The system SHALL expose a `neuralnet ringer migrate` CLI command accepting `--input` and `--output` paths (plus an optional `--overwrite` flag) that copies all required training job artifacts and protects against accidental data loss.

#### Scenario: Migrate a complete legacy training job directory via CLI
- **WHEN** a user runs `neuralnet ringer migrate --input <src> --output <dst>` against a valid legacy training job directory
- **THEN** the command creates `<dst>` containing migrated `config.json`, copied `all_models_results.parquet` and `selected_models.parquet`, and migrated `models/member_*/model.keras` alongside copied `models/member_*/results.json.zip`, without modifying `<src>`

#### Scenario: Refuse in-place migration
- **WHEN** `--input` and `--output` resolve to the same directory
- **THEN** the migration command fails with a non-zero exit code and makes no filesystem changes

#### Scenario: Refuse to overwrite an existing destination directory by default
- **WHEN** `--output` already exists and `--overwrite` is not specified
- **THEN** the migration command fails with a non-zero exit code without modifying existing files

#### Scenario: Allow overwriting an existing destination directory when explicitly requested
- **WHEN** `--output` already exists and `--overwrite` is specified (and `--input` differs from `--output`)
- **THEN** the migration command replaces the destination contents with the newly migrated artifacts

### Requirement: Migrated directory works natively with downstream workflows
The system SHALL ensure that migrated training job directories work natively with `RingerKerasTrainingJob.load()`, `RingerCommitteeThresholdFitJob`, and `BinnedSpecialistCommittee.from_json()` while core loaders and YAML schemas remain strictly validated.

#### Scenario: Run threshold-fit on a migrated legacy training directory
- **WHEN** threshold fitting is executed against a directory produced by `neuralnet ringer migrate`
- **THEN** threshold fitting completes successfully, writes operating-point JSON files and member models, and the resulting committee can be reloaded and used for inference without any legacy runtime shims
