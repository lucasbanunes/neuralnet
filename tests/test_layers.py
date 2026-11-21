import pytest
from pydantic import TypeAdapter, ValidationError

from neuralnet.layers import (
    BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory,
    DenseFactory,
    DenseLayerType,
    FixedPointQuantizedDenseFactory,
    HGQ2DenseFactory,
    KAN1DLayerFactory,
    LayerType,
)
from neuralnet.quantization.hgq import HGQ2MinMaxConstraint, HGQ2QuantizerConfig


def test_dense_factory_rejects_quantization_fields():
    """DenseFactory is strictly floating-point and rejects extra quantization fields."""
    fp = DenseFactory(units=8, activation="relu")
    assert fp.object_type == "dense"

    with pytest.raises(ValidationError):
        DenseFactory(units=8, activation="relu", integer_bits=2, fractional_bits=6)

    with pytest.raises(ValidationError):
        DenseFactory(
            units=8,
            activation="relu",
            quantization=HGQ2QuantizerConfig(i0=2, f0=6),
        )


def test_fixed_point_quantized_dense_factory_rejects_floating_bits():
    """FixedPointQuantizedDenseFactory requires fractional_bits and rejects legacy floating_bits."""
    factory = FixedPointQuantizedDenseFactory(
        units=8,
        activation="relu",
        integer_bits=2,
        fractional_bits=6,
    )
    assert factory.object_type == "fixed_point_quantized_dense"

    with pytest.raises(ValidationError):
        FixedPointQuantizedDenseFactory(
            units=8,
            activation="relu",
            integer_bits=2,
            floating_bits=6,
        )


def test_hgq2_dense_factory_builds_qdense_with_minmax(isolated_executor):
    """Verify HGQ2DenseFactory builds QDense with MinMax constraints for bit optimization."""
    future = isolated_executor.submit(_hgq2_dense_factory_routine)
    future.result()


def _hgq2_dense_factory_routine():
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    from hgq.constraints import MinMax
    from hgq.layers import QDense

    factory = HGQ2DenseFactory(
        units=6,
        activation="relu",
        quantization=HGQ2QuantizerConfig(
            k0=True,
            i0=2,
            f0=6,
            ic=HGQ2MinMaxConstraint(min_value=0, max_value=4),
            fc=HGQ2MinMaxConstraint(min_value=2, max_value=10),
        ),
        name="hgq_hidden",
    )
    layer = factory.as_keras()
    assert isinstance(layer, QDense)
    assert layer.units == 6
    assert layer.name == "hgq_hidden"
    assert isinstance(layer.kq.config.config["ic"], MinMax)
    assert isinstance(layer.kq.config.config["fc"], MinMax)
    assert isinstance(layer.bq.config.config["ic"], MinMax)
    assert isinstance(layer.bq.config.config["fc"], MinMax)


def test_kan1d_layer_factory_schema_and_backend_validation(isolated_executor):
    """Verify KAN1DLayerFactory schema parsing and invalid-backend ValueError."""
    kan = KAN1DLayerFactory(input_size=10, output_size=4, grid_size=5, spline_order=3)
    assert kan.object_type == "kan1d_layer"

    future = isolated_executor.submit(_kan1d_rejects_tf_routine)
    future.result()


def _kan1d_rejects_tf_routine():
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    kan = KAN1DLayerFactory(input_size=4, output_size=2, grid_size=3, spline_order=2)
    with pytest.raises(ValueError, match="KERAS_BACKEND=torch"):
        kan.as_keras()


def test_unified_layer_type_discriminated_union():
    """Parse mixed-layer lists under LayerType and reject unknown object_type."""
    adapter = TypeAdapter(list[LayerType])
    layers = adapter.validate_python(
        [
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
                "units": 4,
                "activation": "tanh",
                "quantization": {
                    "object_type": "hgq2_fixed_point",
                    "i0": 2,
                    "f0": 6,
                    "ic": {"object_type": "min_max", "min_value": 0, "max_value": 4},
                },
            },
            {
                "object_type": "kan1d_layer",
                "input_size": 4,
                "output_size": 2,
                "grid_size": 5,
                "spline_order": 3,
            },
            {
                "object_type": "basic_entangler",
                "n_qubits": 2,
                "n_layers": 2,
            },
        ]
    )
    assert isinstance(layers[0], DenseFactory)
    assert isinstance(layers[1], FixedPointQuantizedDenseFactory)
    assert isinstance(layers[2], HGQ2DenseFactory)
    assert isinstance(layers[3], KAN1DLayerFactory)
    assert isinstance(layers[4], BasicEntanglerQuantumLayerWithAnglerEmbeddingFactory)

    with pytest.raises(ValidationError):
        adapter.validate_python([{"object_type": "unknown_layer", "units": 4}])

    dense_adapter = TypeAdapter(list[DenseLayerType])
    with pytest.raises(ValidationError):
        dense_adapter.validate_python(
            [
                {
                    "object_type": "kan1d_layer",
                    "input_size": 4,
                    "output_size": 2,
                    "grid_size": 5,
                    "spline_order": 3,
                }
            ]
        )
