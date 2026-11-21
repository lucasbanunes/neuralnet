"""Quantization utilities backed by the hgq2 library.

This module contains helpers for converting Keras Sequential models into
HGQ-quantized models using hgq2 as the backend implementation.
"""

from typing import TYPE_CHECKING, Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from hgq.config import QuantizerConfig as HGQLibQuantizerConfig
    from hgq.layers import QDense
    from keras import Sequential
    from keras.layers import Dense


class HGQ2ConstantConstraint(BaseModel):
    """Constant bitwidth constraint for HGQ2 integer or fractional bitwidths."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_type: Annotated[
        Literal["constant"],
        Field(description='Constraint type identifier. Must be "constant".'),
    ] = "constant"
    value: Annotated[
        int,
        Field(description="Constant bitwidth value."),
    ]

    def as_hgq_constraint(self):
        """Convert to an ``hgq.constraints.Constant`` instance."""
        from hgq.constraints import Constant

        return Constant(self.value)


class HGQ2MinMaxConstraint(BaseModel):
    """Learnable min-max bitwidth range constraint for HGQ2 bit optimization."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_type: Annotated[
        Literal["min_max"],
        Field(description='Constraint type identifier. Must be "min_max".'),
    ] = "min_max"
    min_value: Annotated[
        int,
        Field(description="Minimum allowed bitwidth value."),
    ]
    max_value: Annotated[
        int,
        Field(description="Maximum allowed bitwidth value."),
    ]

    @model_validator(mode="after")
    def _validate_bounds(self) -> "HGQ2MinMaxConstraint":
        if self.max_value < self.min_value:
            raise ValueError(
                f"Invalid HGQ2 MinMax constraint: max_value ({self.max_value}) < min_value ({self.min_value})."
            )
        return self

    def as_hgq_constraint(self):
        """Convert to an ``hgq.constraints.MinMax`` instance."""
        from hgq.constraints import MinMax

        return MinMax(self.min_value, self.max_value)


type HGQ2ConstraintType = Annotated[
    HGQ2ConstantConstraint | HGQ2MinMaxConstraint,
    Field(discriminator="object_type"),
]


class HGQ2QuantizerConfig(BaseModel):
    """Fixed-point quantizer configuration using native hgq2 parameters.

    Parameters
    ----------
    k0 : bool, optional
        Whether the fixed-point representation is signed (keep sign bit). Defaults to ``True``.
    i0 : int
        Initial number of integer bits for the fixed-point representation.
    f0 : int
        Initial number of fractional bits for the fixed-point representation.
    ic : HGQ2ConstraintType, optional
        Integer bitwidth constraint. Defaults to ``Constant(i0)`` when omitted.
    fc : HGQ2ConstraintType, optional
        Fractional bitwidth constraint. Defaults to ``Constant(f0)`` when omitted.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_type: Annotated[
        Literal["hgq2_fixed_point"],
        Field(description='Quantizer type identifier. Must be "hgq2_fixed_point".'),
    ] = "hgq2_fixed_point"
    k0: Annotated[
        bool,
        Field(description="Whether to keep the sign bit in the HGQ2 kif representation."),
    ] = True
    i0: Annotated[
        int,
        Field(description="Initial number of integer bits for the HGQ2 fixed-point representation."),
    ]
    f0: Annotated[
        int,
        Field(description="Initial number of fractional bits for the HGQ2 fixed-point representation."),
    ]
    ic: Annotated[
        HGQ2ConstraintType | None,
        Field(description="Optional integer bitwidth constraint (Constant or MinMax). Defaults to Constant(i0)."),
    ] = None
    fc: Annotated[
        HGQ2ConstraintType | None,
        Field(description="Optional fractional bitwidth constraint (Constant or MinMax). Defaults to Constant(f0)."),
    ] = None

    @model_validator(mode="after")
    def _validate_initial_bitwidths(self) -> "HGQ2QuantizerConfig":
        if isinstance(self.ic, HGQ2MinMaxConstraint) and not (self.ic.min_value <= self.i0 <= self.ic.max_value):
            raise ValueError(
                f"Initial integer bitwidth i0 ({self.i0}) must lie within ic range "
                f"[{self.ic.min_value}, {self.ic.max_value}]."
            )
        if isinstance(self.fc, HGQ2MinMaxConstraint) and not (self.fc.min_value <= self.f0 <= self.fc.max_value):
            raise ValueError(
                f"Initial fractional bitwidth f0 ({self.f0}) must lie within fc range "
                f"[{self.fc.min_value}, {self.fc.max_value}]."
            )
        return self

    def as_hgq_quantizer_config(self, place: Literal["weight", "bias"] = "weight") -> "HGQLibQuantizerConfig":
        """Create an hgq2 quantizer configuration from fixed-point settings.

        Parameters
        ----------
        place : {"weight", "bias"}, optional
            Target tensor place for the quantizer configuration. Defaults to ``"weight"``.

        Returns
        -------
        hgq.config.QuantizerConfig
            An hgq2 quantizer configuration using configured integer and
            fractional bitwidth constraints.
        """
        from hgq.config import QuantizerConfig as HGQLibQuantizerConfig
        from hgq.constraints import Constant

        ic_constraint = self.ic.as_hgq_constraint() if self.ic is not None else Constant(self.i0)
        fc_constraint = self.fc.as_hgq_constraint() if self.fc is not None else Constant(self.f0)

        return HGQLibQuantizerConfig(
            q_type="kif",
            place=place,
            k0=self.k0,
            i0=self.i0,
            f0=self.f0,
            ic=ic_constraint,
            fc=fc_constraint,
        )

    def quantize_model(self, model: "Sequential", name: str | None = None) -> "Sequential":
        """Convert a Keras Sequential model into an HGQ2 quantized Sequential model.

        Parameters
        ----------
        model : keras.Sequential
            Source floating-point Sequential model.
        name : str, optional
            Optional name for the quantized model.

        Returns
        -------
        keras.Sequential
            Quantized Sequential model with QDense layers and copied weights.
        """
        return hgq_quantize(
            model,
            weight_quantizer_config=self,
            bias_quantizer_config=self,
            name=name,
        )

    def custom_objects(self) -> dict[str, Any]:
        """Return Keras custom objects needed to deserialize HGQ2 models."""
        from hgq.layers import QDense

        return {"QDense": QDense}


def hgq_quantize(
    model: "Sequential",
    weight_quantizer_config: Union[HGQ2QuantizerConfig, "HGQLibQuantizerConfig"],
    bias_quantizer_config: Union[HGQ2QuantizerConfig, "HGQLibQuantizerConfig"],
    name: str | None = None,
) -> "Sequential":
    """Quantize a Keras Sequential model using hgq2 backend layers.

    Supports Dense and Activation layers.

    Parameters
    ----------
    model : keras.Sequential
        Source Sequential model to convert.
    weight_quantizer_config : HGQ2QuantizerConfig or hgq.config.QuantizerConfig
        Quantizer configuration for the dense layer kernel weights.
    bias_quantizer_config : HGQ2QuantizerConfig or hgq.config.QuantizerConfig
        Quantizer configuration for the dense layer biases.
    name : str, optional
        Name for the quantized Sequential model. If omitted, the new model name will
        be "{old_name}_quantized".

    Returns
    -------
    keras.Sequential
        A new Sequential model whose Dense layers have been replaced by
        hgq2 QDense layers.

    Raises
    ------
    TypeError
        If ``model`` is not a Keras Sequential instance or contains an
        unsupported layer type.
    """
    from keras import Input, Sequential
    from keras.layers import Activation, Dense

    if not isinstance(model, Sequential):
        raise TypeError(f"Expected keras.Sequential model, got {type(model)}")

    if isinstance(weight_quantizer_config, HGQ2QuantizerConfig):
        weight_quantizer_config = weight_quantizer_config.as_hgq_quantizer_config(place="weight")

    if isinstance(bias_quantizer_config, HGQ2QuantizerConfig):
        bias_quantizer_config = bias_quantizer_config.as_hgq_quantizer_config(place="bias")

    quantized_layers = [
        Input(shape=model.input_shape[1:]),
    ]
    for layer in model.layers:
        if isinstance(layer, Dense):
            quantized_layer = keras_dense_to_hgq_dense(
                layer, kq_conf=weight_quantizer_config, bq_conf=bias_quantizer_config
            )
            quantized_layers.append(quantized_layer)
        elif isinstance(layer, Activation):
            quantized_layers.append(layer)
        else:
            raise TypeError(f"Unsupported layer type: {type(layer)}")

    if name is None:
        name = f"{model.name}_quantized"
    return Sequential(quantized_layers, name=name)


def keras_dense_to_hgq_dense(
    keras_layer: "Dense",
    kq_conf: "HGQLibQuantizerConfig | None" = None,
    bq_conf: "HGQLibQuantizerConfig | None" = None,
) -> "QDense":
    """Convert a Keras Dense layer into an HGQ quantized dense layer.

    Parameters
    ----------
    keras_layer : keras.layers.Dense
        Source Keras dense layer whose configuration and weights will be copied.
    kq_conf : hgq.config.QuantizerConfig, optional
        Configuration for the kernel quantizer.
    bq_conf : hgq.config.QuantizerConfig, optional
        Configuration for the bias quantizer.

    Returns
    -------
    hgq.layers.QDense
        A quantized dense layer with the same configuration and copied weights.
    """
    from hgq.layers import QDense

    quantized_layer = QDense(
        units=keras_layer.units,
        activation=keras_layer.activation,
        use_bias=keras_layer.use_bias,
        kernel_initializer=keras_layer.kernel_initializer,
        bias_initializer=keras_layer.bias_initializer,
        kernel_regularizer=keras_layer.kernel_regularizer,
        bias_regularizer=keras_layer.bias_regularizer,
        activity_regularizer=keras_layer.activity_regularizer,
        kernel_constraint=keras_layer.kernel_constraint,
        bias_constraint=keras_layer.bias_constraint,
        kq_conf=kq_conf,
        bq_conf=bq_conf,
    )
    quantized_layer.build((None, int(keras_layer.kernel.shape[0])))
    quantized_layer_weights = quantized_layer.get_weights()
    keras_weights = keras_layer.get_weights()
    quantized_layer_weights[0] = keras_weights[0]  # kernel
    if len(keras_weights) > 1:
        quantized_layer_weights[1] = keras_weights[1]  # bias
    quantized_layer.set_weights(quantized_layer_weights)
    return quantized_layer
