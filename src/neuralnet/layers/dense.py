"""Explicit Keras 3 dense-family layer factories.

This module provides explicit per-layer Pydantic factories for:
- ``DenseFactory`` (``"dense"``): strictly floating-point ``keras.layers.Dense``.
- ``FixedPointQuantizedDenseFactory`` (``"fixed_point_quantized_dense"``): custom fixed-point quantized dense layer.
- ``HGQ2DenseFactory`` (``"hgq2_dense"``): HGQ2 ``QDense`` layer with fixed or learnable bitwidth ranges.
"""

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from neuralnet.quantization.hgq import HGQ2QuantizerConfig
from neuralnet.quantization.quantizers import CustomFixedPointQuantizer

if TYPE_CHECKING:
    from hgq.layers import QDense
    from keras.layers import Dense

    from neuralnet.quantization.keras import FixedPointQuantizationDense


type ActivationField = Annotated[
    str | None,
    Field(
        description="Activation function for the dense layer.",
    ),
]

type BiasInitializerField = Annotated[
    str,
    Field(
        description="Initializer for the bias vector.",
    ),
]

type KernelInitializerField = Annotated[
    str,
    Field(
        description="Initializer for the kernel weights matrix.",
    ),
]

type LayerNameField = Annotated[
    str | None,
    Field(
        description="Name of the layer. Must be a valid Python identifier. If not provided, a default name will be generated.",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    ),
]

type UnitsField = Annotated[
    int,
    Field(
        description="Number of neurons in the dense layer.",
        gt=0,
    ),
]


class DenseFactory(BaseModel):
    """Pydantic factory for a strictly floating-point Keras ``Dense`` layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    units: UnitsField
    activation: ActivationField
    kernel_initializer: KernelInitializerField = "glorot_uniform"
    bias_initializer: BiasInitializerField = "zeros"
    name: LayerNameField = None
    object_type: Annotated[
        Literal["dense"],
        Field(description='Layer type identifier. Must be "dense".'),
    ] = "dense"

    def as_keras(self) -> "Dense":
        """Build a ``keras.layers.Dense`` instance from this factory."""
        from keras.layers import Dense

        return Dense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name=self.name,
        )


class FixedPointQuantizedDenseFactory(BaseModel):
    """Pydantic factory for a single custom ``FixedPointQuantizationDense`` layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    units: UnitsField
    activation: ActivationField
    integer_bits: Annotated[
        int,
        Field(
            description="Number of integer bits for custom fixed-point quantization.",
            gt=0,
        ),
    ]
    fractional_bits: Annotated[
        int,
        Field(
            description="Number of fractional bits for custom fixed-point quantization.",
            gt=0,
        ),
    ]
    kernel_initializer: KernelInitializerField = "glorot_uniform"
    bias_initializer: BiasInitializerField = "zeros"
    name: LayerNameField = None
    object_type: Annotated[
        Literal["fixed_point_quantized_dense"],
        Field(description='Layer type identifier. Must be "fixed_point_quantized_dense".'),
    ] = "fixed_point_quantized_dense"

    @property
    def quantizer(self) -> CustomFixedPointQuantizer:
        """Return the ``CustomFixedPointQuantizer`` configuration for this layer."""
        return CustomFixedPointQuantizer(
            integer_bits=self.integer_bits,
            fractional_bits=self.fractional_bits,
        )

    def as_keras(self) -> "FixedPointQuantizationDense":
        """Build a ``FixedPointQuantizationDense`` instance from this factory."""
        from neuralnet.quantization.keras import FixedPointQuantizationDense

        quantizer = self.quantizer
        return FixedPointQuantizationDense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            fractional_bits=quantizer.fractional_bits,
            integer_bits=quantizer.integer_bits,
            name=self.name,
        )


class HGQ2DenseFactory(BaseModel):
    """Pydantic factory for a single HGQ2 ``QDense`` layer with fixed or learnable bitwidths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    units: UnitsField
    activation: ActivationField
    quantization: Annotated[
        HGQ2QuantizerConfig,
        Field(description="HGQ2 fixed-point quantization configuration (supports Constant and MinMax ranges)."),
    ]
    kernel_initializer: KernelInitializerField = "glorot_uniform"
    bias_initializer: BiasInitializerField = "zeros"
    name: LayerNameField = None
    object_type: Annotated[
        Literal["hgq2_dense"],
        Field(description='Layer type identifier. Must be "hgq2_dense".'),
    ] = "hgq2_dense"

    def as_keras(self) -> "QDense":
        """Build an ``hgq.layers.QDense`` instance from this factory."""
        from hgq.layers import QDense

        return QDense(
            units=self.units,
            activation=self.activation,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            name=self.name,
            kq_conf=self.quantization.as_hgq_quantizer_config(place="weight"),
            bq_conf=self.quantization.as_hgq_quantizer_config(place="bias"),
        )


type DenseLayerType = Annotated[
    DenseFactory | FixedPointQuantizedDenseFactory | HGQ2DenseFactory,
    Field(discriminator="object_type", description="Dense-family layer configuration."),
]
