"""Unified Keras 3 layer factories and discriminated union."""

from typing import Annotated

from pydantic import Field

from neuralnet.layers.dense import (
    DenseFactory,
    DenseLayerType,
    FixedPointQuantizedDenseFactory,
    HGQ2DenseFactory,
)
from neuralnet.layers.kan import KAN1DLayerFactory
from neuralnet.layers.quantum import (
    BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory,
    HardwareEfficientQuantumLayerWithAnglerEmbeddingFactory,
    QuantumLayerFactory,
    StronglyEntanglingQuantumLayerWithAnglerEmbeddingFactory,
)

type LayerType = Annotated[
    (
        DenseFactory
        | FixedPointQuantizedDenseFactory
        | HGQ2DenseFactory
        | KAN1DLayerFactory
        | BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory
        | StronglyEntanglingQuantumLayerWithAnglerEmbeddingFactory
        | HardwareEfficientQuantumLayerWithAnglerEmbeddingFactory
    ),
    Field(discriminator="object_type", description="Keras 3 layer configuration."),
]

__all__ = [
    "BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory",
    "DenseFactory",
    "DenseLayerType",
    "FixedPointQuantizedDenseFactory",
    "HGQ2DenseFactory",
    "HardwareEfficientQuantumLayerWithAnglerEmbeddingFactory",
    "KAN1DLayerFactory",
    "LayerType",
    "QuantumLayerFactory",
    "StronglyEntanglingQuantumLayerWithAnglerEmbeddingFactory",
]
