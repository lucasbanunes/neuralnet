# pluggable-quantization Specification

## Purpose

Provides an extensible, configuration-driven quantization subsystem that supports multiple fixed-point, learnable-range HGQ2, and future quantization backends behind a discriminated-union YAML schema with unified bitwidth nomenclature.

## Requirements

### Requirement: Discriminated-union quantization configuration
The system SHALL represent quantization strategies as validated configuration objects discriminated by `object_type`, supporting at least custom fixed-point quantization (`"custom_fixed_point"`) and HGQ2 quantization (`"hgq2_fixed_point"`), and allowing new quantization strategies to be added without modifying downstream workflow schemas.

#### Scenario: Parse custom fixed-point quantizer from YAML
- **WHEN** a YAML configuration specifies a quantizer with `object_type: "custom_fixed_point"` and positive `integer_bits` and `fractional_bits`
- **THEN** the system validates and instantiates the custom fixed-point quantizer configuration

#### Scenario: Parse HGQ2 fixed-point quantizer from YAML
- **WHEN** a YAML configuration specifies a quantizer with `object_type: "hgq2_fixed_point"` using native `hgq2` parameters (`i0`, `f0`, and optional `k0`, `ic`, `fc`)
- **THEN** the system validates and instantiates the HGQ2 quantizer configuration

#### Scenario: Reject unknown quantizer type
- **WHEN** a YAML configuration specifies an unrecognized `object_type` for a quantizer
- **THEN** schema validation fails with a clear error listing supported quantizer types

### Requirement: Consistent custom fixed-point bitwidth naming
The system SHALL use `integer_bits` and `fractional_bits` exclusively across custom fixed-point quantization configurations for custom Keras layers, post-training model conversion, and tabular feature normalizers, with no backward-compatible aliases for `floating_bits`.

#### Scenario: Configure quantized normalizer with shared fixed-point parameters
- **WHEN** a fixed-point normalizer is configured in YAML using `integer_bits` and `fractional_bits`
- **THEN** the normalizer applies the same representable range and rounding grid as the core fixed-point quantizer

#### Scenario: Reject legacy custom bitwidth field name
- **WHEN** a custom fixed-point quantization or layer YAML configuration uses `floating_bits`
- **THEN** schema validation fails with a validation error requiring `fractional_bits`

#### Scenario: Reject invalid custom fixed-point bitwidths
- **WHEN** a custom fixed-point quantizer configuration specifies `integer_bits <= 0` or `fractional_bits <= 0`
- **THEN** schema validation fails with a validation error

### Requirement: HGQ2 native field names with fixed and learnable bitwidth ranges
The HGQ2 quantization configuration SHALL preserve native `hgq2` parameter names (`k0`, `i0`, `f0`, `ic`, `fc`), where `ic` and `fc` accept either a constant constraint or a min-max bitwidth range (`MinMax`) so HGQ2 layers can optimize bitwidths during training.

#### Scenario: Configure HGQ2 quantizer with fixed bitwidths
- **WHEN** an HGQ2 quantizer configuration specifies `i0` and `f0` with constant constraints (or default constant constraints matching `i0`/`f0`)
- **THEN** the generated HGQ2 quantizer config uses `Constant` constraints at `i0` and `f0`

#### Scenario: Configure HGQ2 quantizer with learnable bitwidth ranges
- **WHEN** an HGQ2 quantizer configuration specifies `MinMax` ranges for `ic` and/or `fc` alongside initial `i0` and `f0`
- **THEN** the generated HGQ2 quantizer config uses `MinMax(min_value, max_value)` constraints with initial bitwidths `i0` and `f0`

#### Scenario: Reject inverted or out-of-bounds HGQ2 bitwidth range
- **WHEN** an HGQ2 `ic` or `fc` min-max constraint specifies `max_value < min_value` or `i0`/`f0` falls outside `[min_value, max_value]`
- **THEN** schema validation fails with a validation error

### Requirement: Post-training model quantization
Each quantization strategy SHALL support converting an existing floating-point sequential model into a quantized sequential model while preserving trained weights.

#### Scenario: Convert floating-point model with custom fixed-point strategy
- **WHEN** post-training quantization is invoked on a sequential dense model with a custom fixed-point quantizer
- **THEN** the system returns a new sequential model with quantized dense layers initialized from the original kernel and bias weights

#### Scenario: Convert floating-point model with HGQ2 strategy
- **WHEN** post-training quantization is invoked on a sequential dense model with an HGQ2 quantizer
- **THEN** the system returns a new sequential model with HGQ2 quantized dense layers initialized from the original kernel and bias weights

### Requirement: Model deserialization custom objects
Each quantization strategy SHALL expose any custom serialization objects required to save and reload quantized models cleanly from disk.

#### Scenario: Save and reload quantized model
- **WHEN** a quantized model built by any supported quantization strategy is saved to a `.keras` file and reloaded through the committee or training loader
- **THEN** the reloaded model produces identical predictions to the model before saving
