# Spec Delta

## Purpose

Establishes a uniform YAML-activity base contract, CLI registration pattern, and single consolidated specialist committee implementation backing the existing Ringer training, threshold-fit, inference, and normalization-analysis workflows.

## ADDED Requirements

### Requirement: Shared BaseActivity YAML contract
All CLI-exposed workflows SHALL inherit from a shared `BaseActivity` base model that enforces strict YAML schema validation (`extra="forbid"`, `frozen=True`), defines standard `output_path` and `logger_name` fields, provides `from_yaml(path)` loading and `dump_config()` snapshot persistence to `output_path/config.json`, and exposes a uniform `submit()` entrypoint.

#### Scenario: Persist resolved configuration on execution
- **WHEN** any activity is loaded from a YAML file and submitted
- **THEN** the activity creates `output_path` if needed and writes the serialized configuration snapshot to `output_path/config.json`

#### Scenario: Reject unknown YAML keys
- **WHEN** an activity YAML file contains an unrecognized field
- **THEN** loading the activity via `from_yaml` fails with a Pydantic validation error

### Requirement: Uniform CLI registration with schema documentation
The CLI SHALL register each YAML activity (`training`, `threshold-fit`, `inference`, `alternative-norm1`) via a shared registration helper that accepts `--config <path>` and renders the activity's Pydantic markdown schema in `--help`.

#### Scenario: Invoke activity help from CLI
- **WHEN** a user runs `neuralnet ringer <activity> --help` for any of the four existing activities
- **THEN** the CLI displays the activity summary and its auto-generated YAML field schema

#### Scenario: Execute activity from YAML config via CLI
- **WHEN** a user runs `neuralnet ringer <activity> --config <yaml-path>` with a valid configuration
- **THEN** the CLI parses the YAML into the activity class and calls `submit()`

### Requirement: Training activity accepts unified model factory union
The Ringer training activity SHALL accept any model factory in the unified model factory union (`keras_sequential`, `mlp`, `kan`) including floating-point, custom fixed-point quantized, and HGQ2 quantized configurations.

#### Scenario: Train HGQ2 quantized MLP from YAML
- **WHEN** a training YAML configures `model_factory` with an MLP or `keras_sequential` using HGQ2 fixed-point quantizers
- **THEN** the training activity validates the YAML, builds the HGQ2 quantized models, trains them, and saves reloadable artifacts

#### Scenario: Train custom fixed-point quantized MLP from YAML
- **WHEN** a training YAML configures `model_factory` with custom fixed-point quantizers
- **THEN** the training activity validates the YAML, builds the custom quantized models, trains them, and saves reloadable artifacts

### Requirement: Single consolidated specialist committee implementation
The system SHALL provide a single canonical `BinnedSpecialistModel` and `BinnedSpecialistCommittee` implementation used by all Ringer workflows and supporting post-training quantization via any pluggable quantization strategy.

#### Scenario: Quantize a specialist committee with any registered strategy
- **WHEN** post-training quantization is applied to a `BinnedSpecialistCommittee` with a pluggable neural-network quantizer configuration (custom fixed-point or HGQ2) and optional normalizer quantizer
- **THEN** every specialist member's model and preprocessing pipeline are quantized using the specified strategy

#### Scenario: Reload standalone operating-point committee JSON
- **WHEN** a standalone operating-point JSON file produced by `threshold-fit` is loaded via `BinnedSpecialistCommittee.from_json`
- **THEN** the committee reconstructs all specialist models (including custom or HGQ2 quantized layers) and preprocessing pipelines without requiring the original training directory
