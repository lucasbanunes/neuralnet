import os

import numpy as np
import polars as pl
import pytest


def make_dataframe(data: dict[str, list[float | int]]) -> pl.DataFrame:
    return pl.DataFrame(data)


def make_lazyframe(data: dict[str, list[float | int]]) -> pl.LazyFrame:
    return pl.DataFrame(data).lazy()


class IdentityTransformation:
    """Minimal Transformation implementation passing through specified feature columns."""

    def __init__(self, cols: list[str]):
        self.input_cols = list(cols)
        self.output_cols = list(cols)

    def __call__(self, data: pl.DataFrame | pl.LazyFrame, passthrough: bool = False):
        if passthrough:
            return data
        return data.select(self.output_cols)


def build_constant_specialist_model():
    os.environ["KERAS_BACKEND"] = "tensorflow"

    import keras

    from neuralnet.bins import VariableBin
    from neuralnet.models.committee import BinnedSpecialistModel

    keras_model = keras.Sequential(
        [
            keras.layers.Input(shape=(2,)),
            keras.layers.Dense(2, activation="relu", use_bias=True),
            keras.layers.Dense(1, activation="sigmoid", use_bias=True),
        ]
    )
    keras_model(np.zeros((1, 2), dtype=np.float32))
    keras_model.layers[0].set_weights(
        [
            np.zeros((2, 2), dtype=np.float32),
            np.array([1.0, 2.0], dtype=np.float32),
        ]
    )
    keras_model.layers[1].set_weights(
        [
            np.zeros((2, 1), dtype=np.float32),
            np.array([np.log(0.7 / 0.3)], dtype=np.float32),
        ]
    )
    bins = [
        VariableBin(var_name="x", low=0.0, high=1.0, closed="both"),
        VariableBin(var_name="y", low=2.0, high=3.0, closed="both"),
    ]
    model = BinnedSpecialistModel(
        bins=bins,
        keras_model=keras_model,
        preprocessing=IdentityTransformation(["x", "y"]),
        decision_threshold=0.5,
    )
    return model


def collect_if_lazy(frame):
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()
    return frame


@pytest.mark.parametrize("frame_factory", [make_dataframe, make_lazyframe])
def test_binned_model_predict_polars_filters_rows_and_adds_predictions(frame_factory, isolated_executor):
    future = isolated_executor.submit(
        binned_model_predict_polars_filters_rows_and_adds_predictions,
        frame_factory=frame_factory,
    )
    future.result()


def binned_model_predict_polars_filters_rows_and_adds_predictions(frame_factory):
    model = build_constant_specialist_model()
    data = frame_factory(
        {
            "id": [1, 2, 3, 4],
            "x": [0.5, 1.5, 0.5, 0.5],
            "y": [2.5, 2.5, 3.5, 2.5],
        }
    )

    result = collect_if_lazy(model.predict_polars(data))

    assert result.columns == ["id", "prediction", "output"]
    assert result.height == 2
    assert result["id"].to_list() == [1, 4]
    assert result["prediction"].to_list() == [True, True]
    for output in result["output"].to_numpy():
        assert output == pytest.approx(0.7, rel=1e-5)


@pytest.mark.parametrize("frame_factory", [make_dataframe, make_lazyframe])
def test_binned_model_predict_polars_with_output_layers_adds_layer_outputs(frame_factory, isolated_executor):
    future = isolated_executor.submit(
        binned_model_predict_polars_with_all_layers_adds_layer_outputs,
        frame_factory=frame_factory,
    )
    future.result()


def binned_model_predict_polars_with_all_layers_adds_layer_outputs(frame_factory):
    model = build_constant_specialist_model()
    data = frame_factory(
        {
            "id": [1, 2, 3, 4],
            "x": [0.5, 1.5, 0.5, 0.5],
            "y": [2.5, 2.5, 3.5, 2.5],
        }
    )

    result = collect_if_lazy(model.predict_polars(data, all_layers=True))

    assert result.columns == [
        "id",
        "prediction",
        "output",
        "layer.dense.0",
        "layer.dense.1",
        "layer.dense_1.0",
    ]
    assert result.height == 2
    assert result["id"].to_list() == [1, 4]
    assert result["prediction"].to_list() == [True, True]
    for output, layer_output in zip(
        result["output"].to_numpy(),
        result["layer.dense_1.0"].to_numpy(),
    ):
        assert output == pytest.approx(0.7, rel=1e-5)
        assert layer_output == pytest.approx(0.7, rel=1e-5)

    assert result["layer.dense.0"].to_list() == [1.0, 1.0]
    assert result["layer.dense.1"].to_list() == [2.0, 2.0]


@pytest.mark.parametrize("frame_factory", [make_dataframe, make_lazyframe])
def test_binned_model_predict_polars_with_all_layers_and_passthrough_adds_layer_outputs(
    frame_factory, isolated_executor
):
    future = isolated_executor.submit(
        binned_model_predict_polars_with_all_layers_and_passthrough_adds_layer_outputs,
        frame_factory=frame_factory,
    )
    future.result()


def binned_model_predict_polars_with_all_layers_and_passthrough_adds_layer_outputs(frame_factory):
    model = build_constant_specialist_model()
    data = frame_factory(
        {
            "id": [1, 2, 3, 4],
            "tag": [10, 20, 30, 40],
            "x": [0.5, 1.5, 0.5, 0.5],
            "y": [2.5, 2.5, 3.5, 2.5],
        }
    )

    result = collect_if_lazy(model.predict_polars(data, all_layers=True, passthrough=True))

    assert result.columns == [
        "id",
        "tag",
        "x",
        "y",
        "prediction",
        "output",
        "layer.dense.0",
        "layer.dense.1",
        "layer.dense_1.0",
    ]
    assert result.height == 4
    assert result["id"].to_list() == [1, 2, 3, 4]
    assert result["tag"].to_list() == [10, 20, 30, 40]
    assert result["prediction"].to_list() == [True, None, None, True]
    assert result["output"].drop_nulls().to_list() == pytest.approx([0.7, 0.7], rel=1e-5)
    assert result["layer.dense.0"].to_list() == [1.0, None, None, 1.0]
    assert result["layer.dense.1"].to_list() == [2.0, None, None, 2.0]
    assert result["layer.dense_1.0"].drop_nulls().to_list() == pytest.approx([0.7, 0.7], rel=1e-5)


def test_binned_specialist_committee_quantize_supports_custom_and_hgq2(isolated_executor):
    future = isolated_executor.submit(_committee_quantize_routine)
    future.result()


def _committee_quantize_routine():
    from hgq.layers import QDense

    from neuralnet.models.committee import BinnedSpecialistCommittee
    from neuralnet.quantization.hgq import HGQ2QuantizerConfig
    from neuralnet.quantization.keras import FixedPointQuantizationDense
    from neuralnet.quantization.quantizers import CustomFixedPointQuantizer

    model = build_constant_specialist_model()
    committee = BinnedSpecialistCommittee(models=[model])

    custom_q = committee.quantize(
        nn_quantizer=CustomFixedPointQuantizer(integer_bits=4, fractional_bits=8),
    )
    assert all(isinstance(layer, FixedPointQuantizationDense) for layer in custom_q.models[0].keras_model.layers)

    hgq_q = committee.quantize(
        nn_quantizer=HGQ2QuantizerConfig(i0=4, f0=8),
    )
    assert all(isinstance(layer, QDense) for layer in hgq_q.models[0].keras_model.layers)
