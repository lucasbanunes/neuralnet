"""Pluggable quantization strategies, configs, and custom-object discovery."""

from neuralnet.quantization.base import (
    KerasQuantizerProtocol,
    PolarsQuantizerProtocol,
    QuantizerConfig,
    get_custom_objects,
)
from neuralnet.quantization.hgq import (
    HGQ2ConstantConstraint,
    HGQ2ConstraintType,
    HGQ2MinMaxConstraint,
    HGQ2QuantizerConfig,
    hgq_quantize,
    keras_dense_to_hgq_dense,
)
from neuralnet.quantization.quantizers import (
    CustomFixedPointQuantizer,
    FixedPointQuantizer,
)

__all__ = [
    "CustomFixedPointQuantizer",
    "FixedPointQuantizer",
    "HGQ2ConstantConstraint",
    "HGQ2ConstraintType",
    "HGQ2MinMaxConstraint",
    "HGQ2QuantizerConfig",
    "KerasQuantizerProtocol",
    "PolarsQuantizerProtocol",
    "QuantizerConfig",
    "get_custom_objects",
    "hgq_quantize",
    "keras_dense_to_hgq_dense",
]
