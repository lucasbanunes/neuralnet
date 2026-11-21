# Spec Delta

## Purpose

Defines a unified, YAML-configurable layer and model factory system for Keras 3 with an explicit factory per layer type, full composition inside a general sequential model factory, and specialized convenience model factories.

## ADDED Requirements

### Requirement: Explicit per-layer factories in a unified discriminated union
The system SHALL provide an explicit factory for each supported Keras 3 layer type under a single discriminated-union layer schema: floating-point dense (`"dense"`), custom fixed-point quantized dense (`"fixed_point_quantized_dense"`), HGQ2 quantized dense (`"hgq2_dense"`), 1D KAN (`"kan1d_layer"`), and PennyLane quantum layers.

#### Scenario: Parse mixed layer list from YAML
- **WHEN** a sequential model YAML configuration lists `"dense"`, `"fixed_point_quantized_dense"`, `"hgq2_dense"`, `"kan1d_layer"`, and quantum layer definitions discriminated by `object_type`
- **THEN** all layer configurations validate cleanly under the unified layer schema

#### Scenario: Keep floating-point dense factory strictly free of quantization fields
- **WHEN** a `"dense"` layer configuration includes any quantization field
- **THEN** schema validation fails because `"dense"` is strictly floating-point and quantized dense layers MUST use their explicit layer factory (`"fixed_point_quantized_dense"` or `"hgq2_dense"`)

#### Scenario: Reject unknown layer type
- **WHEN** a layer definition specifies an unrecognized `object_type`
- **THEN** schema validation fails with a validation error

### Requirement: Explicit quantized dense layer factories with dedicated quantizer logic
The system SHALL provide dedicated `"fixed_point_quantized_dense"` and `"hgq2_dense"` layer factories where `"fixed_point_quantized_dense"` uses its own custom fixed-point quantizer logic and `"hgq2_dense"` uses its own HGQ2 quantizer logic supporting fixed and learnable bitwidth ranges.

#### Scenario: Build custom fixed-point quantized dense layer
- **WHEN** a `"fixed_point_quantized_dense"` layer configuration is built into a Keras layer
- **THEN** the system returns a custom fixed-point quantized dense layer configured with the specified `integer_bits` and `fractional_bits`

#### Scenario: Build HGQ2 quantized dense layer with optimizable bitwidth ranges
- **WHEN** an `"hgq2_dense"` layer configuration specifies learnable bitwidth ranges in its HGQ2 quantization config and is built into a Keras layer
- **THEN** the system returns an HGQ2 `QDense` layer whose kernel and bias quantizers use min-max bitwidth constraints for bit optimization

### Requirement: General sequential model factory
The system SHALL provide a general sequential model factory (`object_type: "keras_sequential"`) that constructs a Keras Sequential model from any ordered list of supported layer configurations.

#### Scenario: Build mixed-layer sequential model
- **WHEN** a `keras_sequential` model configuration contains a `"dense"` layer followed by an `"hgq2_dense"` or `"fixed_point_quantized_dense"` layer
- **THEN** building the model returns a Keras Sequential model with those layers in the configured order

#### Scenario: Require at least one layer
- **WHEN** a `keras_sequential` model configuration provides an empty layer list
- **THEN** schema validation fails with a validation error

### Requirement: Specialized convenience model factories
The system SHALL provide specialized model factories for common homogeneous architectures — including an MLP factory (`object_type: "mlp"`) accepting dense-family layers (`"dense"`, `"fixed_point_quantized_dense"`, `"hgq2_dense"`) and a KAN factory (`object_type: "kan"`) — that compile to standard Keras Sequential models and participate in the same model factory union as the general sequential factory.

#### Scenario: Build floating-point or quantized MLP via MLP factory
- **WHEN** an `mlp` model configuration defines layers using `"dense"`, `"fixed_point_quantized_dense"`, or `"hgq2_dense"`
- **THEN** building the model returns a Keras Sequential model matching the configured layers

#### Scenario: Build multi-layer KAN via KAN factory
- **WHEN** a `kan` model configuration defines a layer-width sequence, grid size, and spline order
- **THEN** building the model returns a Keras Sequential model composed of the corresponding 1D KAN layers

### Requirement: Backend compatibility validation for Torch-wrapped layers
The system SHALL validate that the active Keras backend is compatible before instantiating layers that wrap PyTorch modules (such as 1D KAN and PennyLane quantum layers), failing fast with an actionable error message when the backend is incompatible.

#### Scenario: Instantiate KAN or quantum layer under non-torch backend
- **WHEN** a model containing a KAN 1D or quantum layer is built while the active Keras backend is `tensorflow`
- **THEN** the system raises a clear `ValueError` naming the incompatible layer and stating that `KERAS_BACKEND=torch` is required

#### Scenario: Instantiate KAN or quantum layer under torch backend
- **WHEN** a model containing a KAN 1D or quantum layer is built while the active Keras backend is `torch`
- **THEN** the system builds the Keras Sequential model without backend errors

### Requirement: Single canonical quantum layer module and VKAN deprecation
The system SHALL maintain a single canonical implementation for PennyLane quantum layers and mark the legacy standalone VKAN module as deprecated.

#### Scenario: Import legacy VKAN module
- **WHEN** user code imports the deprecated VKAN module
- **THEN** the system emits a `DeprecationWarning` directing users to the unified KAN factory and Ringer YAML workflows
