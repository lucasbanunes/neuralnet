from pathlib import Path

import pytest
from pydantic import ValidationError

from neuralnet.quantization.hgq import (
    HGQ2ConstantConstraint,
    HGQ2MinMaxConstraint,
    HGQ2QuantizerConfig,
)


def test_hgq2_quantizer_config_fixed_and_minmax_validation():
    """Validate fixed Constant defaults, explicit MinMax ranges, and invalid range rejection."""
    fixed_cfg = HGQ2QuantizerConfig(i0=2, f0=6)
    assert fixed_cfg.object_type == "hgq2_fixed_point"
    assert fixed_cfg.k0 is True
    assert fixed_cfg.ic is None
    assert fixed_cfg.fc is None

    range_cfg = HGQ2QuantizerConfig(
        k0=True,
        i0=2,
        f0=6,
        ic=HGQ2MinMaxConstraint(min_value=0, max_value=4),
        fc=HGQ2MinMaxConstraint(min_value=2, max_value=10),
    )
    assert isinstance(range_cfg.ic, HGQ2MinMaxConstraint)
    assert isinstance(range_cfg.fc, HGQ2MinMaxConstraint)

    const_cfg = HGQ2QuantizerConfig(
        i0=2,
        f0=6,
        ic=HGQ2ConstantConstraint(value=2),
        fc=HGQ2ConstantConstraint(value=6),
    )
    assert isinstance(const_cfg.ic, HGQ2ConstantConstraint)

    # Inverted MinMax range
    with pytest.raises(ValidationError):
        HGQ2MinMaxConstraint(min_value=5, max_value=2)

    # Initial i0 outside MinMax range
    with pytest.raises(ValidationError):
        HGQ2QuantizerConfig(
            i0=5,
            f0=6,
            ic=HGQ2MinMaxConstraint(min_value=0, max_value=4),
        )

    # Initial f0 outside MinMax range
    with pytest.raises(ValidationError):
        HGQ2QuantizerConfig(
            i0=2,
            f0=12,
            fc=HGQ2MinMaxConstraint(min_value=2, max_value=10),
        )


def test_hgq2_as_config_ptq_and_roundtrip(tmp_path: Path, isolated_executor):
    """Verify Constant/MinMax hgq2 config creation, PTQ weight copy, and .keras save/reload."""
    future = isolated_executor.submit(_hgq2_routine, tmp_path=tmp_path)
    future.result()


def _hgq2_routine(tmp_path: Path):
    import os

    os.environ["KERAS_BACKEND"] = "tensorflow"

    import keras
    import numpy as np
    from hgq.constraints import Constant, MinMax
    from hgq.layers import QDense

    from neuralnet.quantization.base import get_custom_objects

    # 1. Fixed bitwidth config -> Constant constraints
    fixed_cfg = HGQ2QuantizerConfig(i0=2, f0=6)
    hgq_fixed = fixed_cfg.as_hgq_quantizer_config(place="weight")
    assert isinstance(hgq_fixed.config["ic"], Constant)
    assert isinstance(hgq_fixed.config["fc"], Constant)

    # 2. MinMax bitwidth config -> MinMax constraints
    range_cfg = HGQ2QuantizerConfig(
        i0=2,
        f0=6,
        ic=HGQ2MinMaxConstraint(min_value=0, max_value=4),
        fc=HGQ2MinMaxConstraint(min_value=2, max_value=10),
    )
    hgq_range = range_cfg.as_hgq_quantizer_config(place="weight")
    assert isinstance(hgq_range.config["ic"], MinMax)
    assert isinstance(hgq_range.config["fc"], MinMax)

    # 3. PTQ weight copy + high precision agreement
    rng = np.random.default_rng(42)
    X = rng.normal(size=(16, 4)).astype(np.float32)

    fp_model = keras.Sequential(
        [
            keras.Input(shape=(4,)),
            keras.layers.Dense(6, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    fp_model.build((None, 4))

    ptq_cfg = HGQ2QuantizerConfig(i0=8, f0=12)
    q_model = ptq_cfg.quantize_model(fp_model)
    assert all(isinstance(layer, QDense) for layer in q_model.layers)

    # Verify weights were copied
    for fp_layer, q_layer in zip(fp_model.layers, q_model.layers):
        np.testing.assert_allclose(q_layer.get_weights()[0], fp_layer.get_weights()[0])
        np.testing.assert_allclose(q_layer.get_weights()[1], fp_layer.get_weights()[1])

    fp_preds = fp_model.predict(X, verbose=0)
    q_preds = q_model.predict(X, verbose=0)
    np.testing.assert_allclose(fp_preds, q_preds, rtol=1e-2, atol=1e-2)

    # 4. .keras save/reload round-trip using get_custom_objects()
    model_path = tmp_path / "hgq2_model.keras"
    q_model.save(model_path)
    loaded = keras.saving.load_model(model_path, custom_objects=get_custom_objects())
    loaded_preds = loaded.predict(X, verbose=0)
    np.testing.assert_allclose(q_preds, loaded_preds, rtol=1e-5, atol=1e-5)
