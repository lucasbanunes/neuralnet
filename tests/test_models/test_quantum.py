import numpy as np
import pytest


@pytest.mark.parametrize(
    ("class_name", "kwargs", "name"),
    [
        (
            "BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory",
            {
                "n_qubits": 2,
                "n_layers": 2,
                "name": "basic_entangler",
                "shots": None,
                "diff_method": "backprop",
            },
            "basic_entangler",
        ),
        (
            "StronglyEntanglingQuantumLayerWithAnglerEmbeddingFactory",
            {
                "n_qubits": 3,
                "n_layers": 1,
                "name": "strongly_entangling",
                "shots": 32,
                "diff_method": "parameter-shift",
            },
            "strongly_entangling",
        ),
        (
            "HardwareEfficientQuantumLayerWithAnglerEmbeddingFactory",
            {
                "n_qubits": 4,
                "n_layers": 2,
                "name": "hardware_efficient",
                "shots": 128,
                "diff_method": "finite-diff",
            },
            "hardware_efficient",
        ),
    ],
)
def test_quantum_layer_config_returns_torch_module_wrapper(class_name: str, kwargs: dict, name: str, isolated_executor):
    future = isolated_executor.submit(
        quantum_layer_config_returns_torch_module_wrapper,
        class_name=class_name,
        kwargs=kwargs,
        name=name,
    )
    future.result()


def quantum_layer_config_returns_torch_module_wrapper(
    class_name: str,
    kwargs: dict,
    name: str,
):
    import os

    os.environ["KERAS_BACKEND"] = "torch"
    import keras

    from neuralnet.layers import quantum

    config_cls = getattr(quantum, class_name)
    config: quantum.QuantumLayerFactory = config_cls(**kwargs)

    assert config.diff_method == kwargs.get("diff_method", "backprop")
    assert config.shots == kwargs.get("shots")
    assert config.name == name

    layer = config.as_keras()

    assert isinstance(layer, keras.layers.TorchModuleWrapper), (
        f"Expected instance of 'TorchModuleWrapper', got '{type(layer).__name__}'"
    )
    assert str(layer.output_shape) == str(kwargs.get("output_shape", (None, kwargs["n_qubits"])))
    assert layer.name == kwargs["name"]

    batch = 2
    inp = np.random.RandomState(42).randn(batch, kwargs["n_qubits"]).astype(np.float32)
    out = layer(inp)

    out_np = out.detach().cpu().numpy() if hasattr(out, "detach") else np.array(out)

    expected_dim = kwargs.get("output_shape", (None, kwargs["n_qubits"]))[1]
    assert out_np.shape == (batch, expected_dim)


def test_quantum_layer_rejects_tensorflow_backend(isolated_executor):
    future = isolated_executor.submit(_quantum_layer_rejects_tf_routine)
    future.result()


def _quantum_layer_rejects_tf_routine():
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    from neuralnet.layers.quantum import BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory

    factory = BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory(n_qubits=2, n_layers=1)
    with pytest.raises(ValueError, match="KERAS_BACKEND=torch"):
        factory.as_keras()
