"""Specialized convenience KAN model factory compiling to a Keras Sequential model."""

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from neuralnet.layers.kan import GridSizeType, KAN1DLayerFactory, SplineOrderType
from neuralnet.models.sequential import NameType

if TYPE_CHECKING:
    from keras import Sequential


type ModelShapeElementType = Annotated[
    int,
    Field(gt=0),
]


class KANFactory(BaseModel):
    """Pydantic factory for a multi-layer 1D KAN model compiled to ``keras.Sequential``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_shape: Annotated[
        list[ModelShapeElementType],
        Field(
            min_length=2,
            description=(
                "The shape of the KAN model, defined as a list of layer sizes "
                "starting with the input size and ending with the output size."
            ),
        ),
    ]
    grid_size: GridSizeType
    spline_order: SplineOrderType
    name: NameType = "kan"
    object_type: Annotated[
        Literal["kan"],
        Field(description='Model factory identifier. Must be "kan".'),
    ] = "kan"

    def as_torch(self):
        """Build an ``efficient_kan.KAN`` model for PyTorch."""
        from efficient_kan import KAN

        return KAN(
            self.model_shape,
            grid_size=self.grid_size,
            spline_order=self.spline_order,
        )

    def as_keras(self) -> "Sequential":
        """Build a ``keras.Sequential`` model composed of ``KAN1DLayerFactory`` layers."""
        from keras import Input, Sequential

        kan_layers = [
            KAN1DLayerFactory(
                input_size=in_dim,
                output_size=out_dim,
                grid_size=self.grid_size,
                spline_order=self.spline_order,
            ).as_keras()
            for in_dim, out_dim in zip(self.model_shape[:-1], self.model_shape[1:])
        ]
        return Sequential(
            [Input(shape=(self.model_shape[0],)), *kan_layers],
            name=self.name,
        )
