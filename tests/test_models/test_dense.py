import os

import numpy as np
import pytest
from pydantic import TypeAdapter, ValidationError


def test_hgqdense_from_keras_dense_copies_configuration_and_weights(isolated_executor):
    future = isolated_executor.submit(
        hgqdense_from_keras_dense_copies_configuration_and_weights,
    )
    future.result()


def hgqdense_from_keras_dense_copies_configuration_and_weights():
    os.environ["KERAS_BACKEND"] = "tensorflow"

    import keras
    from hgq.layers import QDense

    from neuralnet.quantization.hgq import keras_dense_to_hgq_dense

    keras_layer = keras.layers.Dense(
        units=3,
        activation="relu",
        use_bias=True,
        kernel_initializer="zeros",
        bias_initializer="zeros",
    )
    keras_layer.build((None, 4))
    expected_weights = [
        np.arange(12, dtype=np.float32).reshape(4, 3),
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
    ]
    keras_layer.set_weights(expected_weights)

    quantized_layer = keras_dense_to_hgq_dense(keras_layer)

    assert isinstance(quantized_layer, QDense)
    assert quantized_layer.units == keras_layer.units
    assert quantized_layer.activation == keras_layer.activation
    assert quantized_layer.use_bias is True
    quantized_weights = quantized_layer.get_weights()
    np.testing.assert_allclose(quantized_weights[0], expected_weights[0])
    np.testing.assert_allclose(quantized_weights[1], expected_weights[1])


def test_fixed_point_quantization_dense_layer_as_keras(isolated_executor):
    future = isolated_executor.submit(_fixed_point_quantization_dense_layer_as_keras_routine)
    future.result()


def _fixed_point_quantization_dense_layer_as_keras_routine():
    os.environ["KERAS_BACKEND"] = "tensorflow"

    from neuralnet.layers.dense import FixedPointQuantizedDenseFactory
    from neuralnet.quantization.keras import FixedPointQuantizationDense

    factory = FixedPointQuantizedDenseFactory(
        units=4,
        activation="relu",
        kernel_initializer="ones",
        bias_initializer="ones",
        fractional_bits=8,
        integer_bits=4,
        name="test_fp_dense",
    )
    layer = factory.as_keras()

    assert isinstance(layer, FixedPointQuantizationDense)
    assert layer.units == 4
    assert layer.activation == "relu"
    assert layer.name == "test_fp_dense"

    layer.build((None, 2))
    np.testing.assert_allclose(layer.get_weights()[0], np.ones((2, 4), dtype=np.float32))
    np.testing.assert_allclose(layer.get_weights()[1], np.ones((4,), dtype=np.float32))


def test_mlp_and_sequential_model_factories(isolated_executor):
    future = isolated_executor.submit(_mlp_and_sequential_routine)
    future.result()


def _mlp_and_sequential_routine():
    os.environ["KERAS_BACKEND"] = "tensorflow"

    import keras
    from hgq.layers import QDense

    from neuralnet.models import KerasSequentialModelFactory, MLPFactory, ModelFactoryType
    from neuralnet.quantization.keras import FixedPointQuantizationDense

    adapter = TypeAdapter(ModelFactoryType)

    # 1. Mixed dense-family MLP via MLPFactory
    mlp_factory = adapter.validate_python(
        {
            "object_type": "mlp",
            "name": "mixed_mlp",
            "layers": [
                {"object_type": "dense", "units": 8, "activation": "relu"},
                {
                    "object_type": "fixed_point_quantized_dense",
                    "units": 4,
                    "activation": "relu",
                    "integer_bits": 2,
                    "fractional_bits": 6,
                },
                {
                    "object_type": "hgq2_dense",
                    "units": 1,
                    "activation": "sigmoid",
                    "quantization": {"object_type": "hgq2_fixed_point", "i0": 2, "f0": 6},
                },
            ],
        }
    )
    assert isinstance(mlp_factory, MLPFactory)
    mlp_model = mlp_factory.as_keras()
    assert isinstance(mlp_model, keras.Sequential)
    assert isinstance(mlp_model.layers[0], keras.layers.Dense)
    assert isinstance(mlp_model.layers[1], FixedPointQuantizationDense)
    assert isinstance(mlp_model.layers[2], QDense)

    # 2. General KerasSequentialModelFactory
    seq_factory = adapter.validate_python(
        {
            "object_type": "keras_sequential",
            "name": "seq_model",
            "layers": [
                {"object_type": "dense", "units": 4, "activation": "relu"},
                {
                    "object_type": "hgq2_dense",
                    "units": 1,
                    "activation": "sigmoid",
                    "quantization": {"object_type": "hgq2_fixed_point", "i0": 2, "f0": 6},
                },
            ],
        }
    )
    assert isinstance(seq_factory, KerasSequentialModelFactory)
    seq_model = seq_factory.as_keras()
    assert isinstance(seq_model, keras.Sequential)
    assert len(seq_model.layers) == 2

    # 3. Empty layer list rejected
    with pytest.raises(ValidationError):
        KerasSequentialModelFactory(layers=[])

    with pytest.raises(ValidationError):
        MLPFactory(layers=[])
