import importlib
import sys
import warnings

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError

from neuralnet.models import KANFactory, ModelFactoryType


def test_kan_factory_schema_validation():
    """Validate KANFactory schema parsing through ModelFactoryType."""
    adapter = TypeAdapter(ModelFactoryType)
    factory = adapter.validate_python(
        {
            "object_type": "kan",
            "model_shape": [10, 5, 1],
            "grid_size": 5,
            "spline_order": 3,
            "name": "test_kan",
        }
    )
    assert isinstance(factory, KANFactory)
    assert factory.model_shape == [10, 5, 1]

    with pytest.raises(ValidationError):
        KANFactory(model_shape=[10], grid_size=5, spline_order=3)


def test_kan_factory_builds_keras_sequential_under_torch(isolated_executor):
    """Build a multi-layer KAN as a Keras Sequential model under the torch backend."""
    future = isolated_executor.submit(_kan_factory_torch_routine)
    future.result()


def _kan_factory_torch_routine():
    import os

    os.environ["KERAS_BACKEND"] = "torch"

    import keras

    factory = KANFactory(model_shape=[4, 3, 1], grid_size=3, spline_order=2, name="kan_seq")
    model = factory.as_keras()
    assert isinstance(model, keras.Sequential)
    assert len(model.layers) == 2

    x = np.zeros((2, 4), dtype=np.float32)
    out = model(x)
    out_np = out.detach().cpu().numpy() if hasattr(out, "detach") else np.asarray(out)
    assert out_np.shape == (2, 1)


def test_kan_factory_rejects_tensorflow_backend(isolated_executor):
    """Ensure KANFactory raises an actionable ValueError under tensorflow backend."""
    future = isolated_executor.submit(_kan_factory_tf_routine)
    future.result()


def _kan_factory_tf_routine():
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    factory = KANFactory(model_shape=[4, 3, 1], grid_size=3, spline_order=2)
    with pytest.raises(ValueError, match="KERAS_BACKEND=torch"):
        factory.as_keras()


def test_vkan_module_emits_deprecation_warning():
    """Ensure importing neuralnet.models.vkan emits a DeprecationWarning."""
    sys.modules.pop("neuralnet.models.vkan", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        importlib.import_module("neuralnet.models.vkan")

    dep_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("KANFactory" in str(w.message) for w in dep_warnings)
