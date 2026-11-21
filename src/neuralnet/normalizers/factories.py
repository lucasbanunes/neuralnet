from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from neuralnet.quantization.quantizers import FixedPointQuantizer

from .polars import FixedPointQuantizedAlternativeNorm1


class FixedPointAlternativeNormL1(FixedPointQuantizer):
    """YAML factory for fixed-point quantized AlternativeNorm1 preprocessing."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_type: Annotated[
        Literal["fixed_point_alternative_norm1"],
        Field(description='Normalizer type identifier. Must be "fixed_point_alternative_norm1".'),
    ] = "fixed_point_alternative_norm1"

    def as_polars_transform(self, input_cols: list[str]) -> FixedPointQuantizedAlternativeNorm1:
        """Create a FixedPointQuantizedAlternativeNorm1 instance from the configuration.

        Returns
        -------
        FixedPointQuantizedAlternativeNorm1
            An instance of FixedPointQuantizedAlternativeNorm1 with the specified integer and fractional bits.
        """
        return FixedPointQuantizedAlternativeNorm1(
            input_cols=input_cols,
            quantizer=FixedPointQuantizer(
                integer_bits=self.integer_bits,
                fractional_bits=self.fractional_bits,
            ),
        )
