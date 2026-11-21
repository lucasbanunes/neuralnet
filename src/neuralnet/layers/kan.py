"""Factory model for building 1D KAN layers from validated configs."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

type GridSizeType = Annotated[
    int,
    Field(
        description="The size of the grid for the KAN.",
        gt=0,
    ),
]

type SplineOrderType = Annotated[
    int,
    Field(
        description="The order of the spline used for the KAN.",
        gt=0,
    ),
]


class KAN1DLayerFactory(BaseModel):
    """Pydantic factory for a single 1D KAN layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    input_size: Annotated[int, Field(description="The input size.", gt=0)]
    output_size: Annotated[int, Field(description="The output size.", gt=0)]
    grid_size: GridSizeType
    spline_order: SplineOrderType
    name: Annotated[
        str | None,
        Field(description="The name of the layer."),
    ] = None
    object_type: Annotated[
        Literal["kan1d_layer"],
        Field(description='Layer type identifier. Must be "kan1d_layer".'),
    ] = "kan1d_layer"

    def as_torch(self):
        """Build an ``efficient_kan.KAN`` layer for PyTorch.

        Returns
        -------
        efficient_kan.KAN
            KAN module configured as a single layer transform.

        Raises
        ------
        ValueError
            If ``name`` is provided, since naming is not supported for the
            generated PyTorch layer.
        """
        from efficient_kan import KAN

        if self.name is not None:
            raise ValueError("Naming is not supported for PyTorch layers.")

        return KAN(
            [self.input_size, self.output_size],
            grid_size=self.grid_size,
            spline_order=self.spline_order,
        )

    def as_keras(self):
        """Build a Keras wrapper for the generated PyTorch KAN layer.

        Returns
        -------
        keras.layers.TorchModuleWrapper
            Keras layer wrapping the underlying PyTorch KAN module.

        Raises
        ------
        ValueError
            If the active Keras backend is not ``torch``.
        """
        import keras
        from keras.layers import TorchModuleWrapper

        backend = keras.config.backend()
        if backend != "torch":
            raise ValueError(
                f"Incompatible Keras backend '{backend}' for layer 'KAN1DLayerFactory' "
                f"(object_type='{self.object_type}'). Set KERAS_BACKEND=torch to use KAN 1D layers."
            )
        from efficient_kan import KAN

        torch_layer = KAN(
            [self.input_size, self.output_size],
            grid_size=self.grid_size,
            spline_order=self.spline_order,
        )
        return TorchModuleWrapper(
            torch_layer,
            output_shape=[None, self.output_size],
            name=self.name,
        )
