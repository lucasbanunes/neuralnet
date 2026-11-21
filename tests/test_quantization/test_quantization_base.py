import pytest
from pydantic import TypeAdapter, ValidationError

from neuralnet.quantization.base import (
    KerasQuantizerProtocol,
    PolarsQuantizerProtocol,
    QuantizerConfig,
    get_custom_objects,
)
from neuralnet.quantization.hgq import HGQ2QuantizerConfig
from neuralnet.quantization.quantizers import CustomFixedPointQuantizer


def test_quantizer_protocol_compliance():
    """Confirm concrete quantizers satisfy the Keras and Polars protocols."""
    custom = CustomFixedPointQuantizer(integer_bits=2, fractional_bits=6)
    hgq = HGQ2QuantizerConfig(i0=2, f0=6)

    assert isinstance(custom, KerasQuantizerProtocol)
    assert isinstance(custom, PolarsQuantizerProtocol)
    assert isinstance(hgq, KerasQuantizerProtocol)


def test_quantizer_config_discriminated_union():
    """Parse both quantizer types through QuantizerConfig and reject unknown types."""
    adapter = TypeAdapter(QuantizerConfig)

    custom = adapter.validate_python({"object_type": "custom_fixed_point", "integer_bits": 2, "fractional_bits": 6})
    assert isinstance(custom, CustomFixedPointQuantizer)
    assert custom.integer_bits == 2
    assert custom.fractional_bits == 6

    hgq = adapter.validate_python({"object_type": "hgq2_fixed_point", "i0": 2, "f0": 6})
    assert isinstance(hgq, HGQ2QuantizerConfig)
    assert hgq.k0 is True
    assert hgq.i0 == 2
    assert hgq.f0 == 6

    with pytest.raises(ValidationError):
        adapter.validate_python({"object_type": "unknown_quantizer", "i0": 2, "f0": 6})


def test_get_custom_objects_discovers_registered_layers(isolated_executor):
    """Confirm get_custom_objects discovers FixedPointQuantizationDense and QDense."""
    future = isolated_executor.submit(_check_custom_objects)
    future.result()


def _check_custom_objects():
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    objs = get_custom_objects()
    assert "FixedPointQuantizationDense" in objs
    assert "QDense" in objs
