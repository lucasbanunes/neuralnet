"""General Keras 3 Sequential model factory accepting any supported layer."""

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from neuralnet.layers import LayerType

if TYPE_CHECKING:
    from keras import Sequential


type NameType = Annotated[
    str,
    Field(
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
        description="Name of the model. Must be a valid Python identifier.",
    ),
]


class KerasSequentialModelFactory(BaseModel):
    """Pydantic factory for a general ``keras.Sequential`` model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layers: Annotated[
        list[LayerType],
        Field(min_length=1, description="Ordered list of Keras 3 layer configurations."),
    ]
    name: NameType = "keras_sequential"
    object_type: Annotated[
        Literal["keras_sequential"],
        Field(description='Model factory identifier. Must be "keras_sequential".'),
    ] = "keras_sequential"

    def as_keras(self) -> "Sequential":
        """Build a ``keras.Sequential`` model from the configured layers."""
        from keras import Sequential

        return Sequential([layer.as_keras() for layer in self.layers], name=self.name)
