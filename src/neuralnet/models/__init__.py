"""Unified Keras 3 model factories and canonical specialist committee."""

from typing import Annotated

from pydantic import Field

from neuralnet.models.committee import (
    BinnedSpecialistCommittee,
    BinnedSpecialistModel,
    NormStrategyType,
    OperatingPointReferenceConfig,
    SpecialistBinConfig,
    SpecialistCommitteeConfig,
    SpecialistMemberConfig,
    SpecialistPreprocessingConfig,
    Transformation,
)
from neuralnet.models.kan import KANFactory
from neuralnet.models.mlp import MLPFactory
from neuralnet.models.sequential import KerasSequentialModelFactory

type ModelFactoryType = Annotated[
    KerasSequentialModelFactory | MLPFactory | KANFactory,
    Field(discriminator="object_type", description="Keras 3 model factory configuration."),
]

__all__ = [
    "BinnedSpecialistCommittee",
    "BinnedSpecialistModel",
    "KANFactory",
    "KerasSequentialModelFactory",
    "MLPFactory",
    "ModelFactoryType",
    "NormStrategyType",
    "OperatingPointReferenceConfig",
    "SpecialistBinConfig",
    "SpecialistCommitteeConfig",
    "SpecialistMemberConfig",
    "SpecialistPreprocessingConfig",
    "Transformation",
]
