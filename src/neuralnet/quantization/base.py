"""Base protocols, discriminated union, and custom-object discovery for quantization."""

from typing import TYPE_CHECKING, Annotated, Any, Protocol, runtime_checkable

import polars as pl
from pydantic import Field

from neuralnet.quantization.hgq import HGQ2QuantizerConfig
from neuralnet.quantization.quantizers import CustomFixedPointQuantizer

if TYPE_CHECKING:
    from keras import Sequential


@runtime_checkable
class KerasQuantizerProtocol(Protocol):
    """Protocol implemented by Keras-capable quantization strategy configs."""

    object_type: str

    def quantize_model(self, model: "Sequential", name: str | None = None) -> "Sequential":
        """Convert a Keras Sequential model into a quantized Sequential model."""
        ...

    def custom_objects(self) -> dict[str, Any]:
        """Return custom Keras serialization objects needed to reload quantized models."""
        ...


@runtime_checkable
class PolarsQuantizerProtocol(Protocol):
    """Protocol implemented by Polars-capable quantization strategy configs."""

    def quantize_polars_expr(self, expr: pl.Expr) -> pl.Expr:
        """Quantize a Polars expression to the configured representable grid."""
        ...


type QuantizerConfig = Annotated[
    CustomFixedPointQuantizer | HGQ2QuantizerConfig,
    Field(discriminator="object_type"),
]


def get_custom_objects() -> dict[str, Any]:
    """Aggregate Keras custom objects across all registered quantization strategies.

    Lazily imports optional quantization backends (such as ``hgq2``) so that
    calling ``keras.saving.load_model(..., custom_objects=get_custom_objects())``
    seamlessly deserializes models produced by any supported strategy.
    """
    from neuralnet.quantization.keras import FixedPointQuantizationDense

    custom_objects: dict[str, Any] = {
        "FixedPointQuantizationDense": FixedPointQuantizationDense,
    }
    try:
        from hgq.layers import QDense

        custom_objects["QDense"] = QDense
    except ImportError:
        pass

    return custom_objects
