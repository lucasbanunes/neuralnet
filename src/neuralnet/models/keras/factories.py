"""Shared Keras training factory types and sequential model factory re-exports."""

from datetime import datetime
from typing import Annotated, Literal, TypedDict

from pydantic import Field

from neuralnet.layers import LayerType
from neuralnet.losses.binary_loss import BinaryCrossEntropyLossFactory
from neuralnet.models.sequential import KerasSequentialModelFactory, NameType
from neuralnet.optimizers.adam import AdamFactory

EpochsType = Annotated[
    int,
    Field(description="Number of epochs to train the model.", gt=0),
]

type VerboseType = Annotated[
    int,
    Field(ge=0, le=2, description="Verbosity of the training."),
]

type OptimizerType = Annotated[
    AdamFactory,
    Field(discriminator="object_type", description="Optimizer configuration field."),
]

type LossType = Annotated[
    BinaryCrossEntropyLossFactory,
    Field(discriminator="object_type", description="Loss configuration field."),
]

type JitCompileType = Annotated[
    Literal["auto"] | bool,
    Field(
        "auto",
        description="Whether to use JIT compilation for the model.",
    ),
]

type LayersType = Annotated[
    list[LayerType],
    Field(min_length=1, description="List of model layers."),
]


class StandardEvaluationDict(TypedDict):
    loss: float
    start: datetime
    end: datetime


class FitDict(TypedDict):
    loss: list[float]


class FitRoutineDict(TypedDict):
    train: FitDict
    val: FitDict
    start: datetime
    end: datetime


__all__ = [
    "EpochsType",
    "FitDict",
    "FitRoutineDict",
    "JitCompileType",
    "KerasSequentialModelFactory",
    "LayerType",
    "LayersType",
    "LossType",
    "NameType",
    "OptimizerType",
    "StandardEvaluationDict",
    "VerboseType",
]
