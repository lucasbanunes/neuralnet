"""Specialized convenience MLP model factory accepting dense-family layers."""

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from neuralnet.layers.dense import DenseLayerType
from neuralnet.models.sequential import NameType

if TYPE_CHECKING:
    from keras import Sequential


class MLPFactory(BaseModel):
    """Pydantic factory for a Sequential multi-layer perceptron model.

    Accepts any combination of dense-family layers (``"dense"``,
    ``"fixed_point_quantized_dense"``, and ``"hgq2_dense"``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    layers: Annotated[
        list[DenseLayerType],
        Field(min_length=1, description="List of dense-family layers in the MLP."),
    ]
    name: NameType = "mlp"
    object_type: Annotated[
        Literal["mlp"],
        Field(description='Model factory identifier. Must be "mlp".'),
    ] = "mlp"

    def as_keras(self) -> "Sequential":
        """Build a ``keras.Sequential`` MLP model from this factory."""
        from keras import Sequential

        return Sequential(
            [layer.as_keras() for layer in self.layers],
            name=self.name,
        )
