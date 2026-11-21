"""Quantizer models and helpers for quantization methods.

This module defines reusable quantizer configuration classes
and convenience helpers for applying fixed-point quantization.
"""

from functools import cached_property
from typing import TYPE_CHECKING, Annotated, Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from keras import Sequential


class FixedPointQuantizer(BaseModel):
    """Configuration for fixed-point quantization of numeric values."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    integer_bits: Annotated[
        int,
        Field(
            gt=0,
            description="Number of integer bits for fixed-point quantization.",
        ),
    ] = 1
    fractional_bits: Annotated[
        int,
        Field(
            gt=0,
            description="Number of fractional bits for fixed-point quantization.",
        ),
    ]

    @cached_property
    def lower_bound(self) -> float:
        """Return the minimum representable fixed-point value.

        Returns
        -------
        float
            Lower clipping bound used before quantization.
        """
        return -(2 ** (self.integer_bits))

    @cached_property
    def upper_bound(self) -> float:
        """Return the maximum representable fixed-point value.

        Returns
        -------
        float
            Upper clipping bound used before quantization.
        """
        return (2**self.integer_bits) - (2**-self.fractional_bits)

    @cached_property
    def floating_power(self) -> float:
        """Return the power-of-two scaling factor used for quantization.

        Returns
        -------
        float
            Scaling factor equal to :math:`2^{\\mathrm{fractional\\_bits}}`.
        """
        return 2**self.fractional_bits

    def quantize_polars_expr(self, expr: pl.Expr) -> pl.Expr:
        """Quantize a Polars expression to the configured fixed-point grid.

        Parameters
        ----------
        expr : polars.Expr
            Expression whose values should be clipped and quantized.

        Returns
        -------
        polars.Expr
            Expression that evaluates to fixed-point quantized values.
        """
        return (
            expr.mul(self.floating_power).round().truediv(self.floating_power).clip(self.lower_bound, self.upper_bound)
        )


class CustomFixedPointQuantizer(FixedPointQuantizer):
    """Custom fixed-point quantization strategy for both Polars and Keras models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_type: Annotated[
        Literal["custom_fixed_point"],
        Field(description='Quantizer type identifier. Must be "custom_fixed_point".'),
    ] = "custom_fixed_point"

    def quantize_model(self, model: "Sequential", name: str | None = None) -> "Sequential":
        """Convert a Keras Sequential model into a custom fixed-point quantized model.

        Parameters
        ----------
        model : keras.Sequential
            Source floating-point Sequential model.
        name : str, optional
            Optional name for the quantized model.

        Returns
        -------
        keras.Sequential
            Quantized Sequential model with copied weights.
        """
        from neuralnet.quantization.keras import fixed_point_quantize

        return fixed_point_quantize(
            model,
            integer_bits=self.integer_bits,
            fractional_bits=self.fractional_bits,
            name=name,
        )

    def custom_objects(self) -> dict[str, Any]:
        """Return Keras custom objects needed to deserialize custom quantized models."""
        from neuralnet.quantization.keras import FixedPointQuantizationDense

        return {"FixedPointQuantizationDense": FixedPointQuantizationDense}
