import os
from pathlib import Path

import numpy as np
import polars as pl

from neuralnet.bins import VariableBin
from neuralnet.models.committee import BinnedSpecialistCommittee, BinnedSpecialistModel
from neuralnet.quantization.keras import FixedPointQuantizationDense
from neuralnet.quantization.quantizers import FixedPointQuantizer


RINGS_COL = "rings"
RING_FRACTION = 2
N_RINGS = 100  # total rings in the dummy data (> 8+64+8+8+4+4+4 = 100)
N_SAMPLES = 20


def make_dummy_data() -> pl.DataFrame:
    """Build a minimal Polars DataFrame with rings, et, eta and id columns."""
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            "id": np.arange(N_SAMPLES, dtype=np.int64),
            RINGS_COL: [rng.random(N_RINGS, dtype=np.float32).tolist() for _ in range(N_SAMPLES)],
            "et": np.linspace(0.0, 99.0, N_SAMPLES, dtype=np.float32),
            "eta": np.linspace(0.0, 2.4, N_SAMPLES, dtype=np.float32),
        }
    )


def build_preprocessing_pipeline():
    """Build the actual PreprocessingPipeline used in training.py."""
    from neuralnet.workflows.ringer.training import PreprocessingPipeline

    return PreprocessingPipeline.from_job_params(
        rings_col=RINGS_COL,
        ring_fraction=RING_FRACTION,
        norm_strategy="l1",
    )


def build_test_binned_specialist_model():
    os.environ["KERAS_BACKEND"] = "tensorflow"
    import keras

    preprocessing = build_preprocessing_pipeline()
    n_features = len(preprocessing.output_cols)

    keras_model = keras.Sequential(
        [
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(4, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    keras_model(np.zeros((1, n_features), dtype=np.float32))

    bins = [VariableBin(var_name="et", low=0.0, high=100.0, closed="left")]

    return BinnedSpecialistModel(
        bins=bins,
        keras_model=keras_model,
        preprocessing=preprocessing,
        decision_threshold=0.5,
        fold=0,
        training_results={"id": 1, "score": 0.95},
    )


def test_binned_specialist_model_fixed_point_quantization():
    model = build_test_binned_specialist_model()
    norm_quantizer = FixedPointQuantizer(integer_bits=1, fractional_bits=7)
    nn_quantizer = FixedPointQuantizer(integer_bits=2, fractional_bits=6)

    quantized_model = model.fixed_point_quantization(
        norm_quantizer=norm_quantizer,
        nn_quantizer=nn_quantizer,
    )

    assert isinstance(quantized_model, BinnedSpecialistModel)
    assert quantized_model.fold == model.fold
    assert quantized_model.decision_threshold == model.decision_threshold
    assert quantized_model.training_results == model.training_results

    # Preprocessing should have been quantized to FixedPointQuantizedAlternativeNorm1
    from neuralnet.normalizers.polars import FixedPointQuantizedAlternativeNorm1

    assert isinstance(quantized_model.preprocessing.normalizer, FixedPointQuantizedAlternativeNorm1)

    # Dense layers should be converted to FixedPointQuantizationDense
    dense_layers = [l for l in quantized_model.keras_model.layers if isinstance(l, FixedPointQuantizationDense)]
    assert len(dense_layers) == 2


def test_binned_specialist_model_quantization_predict_runs():
    """Verify the quantized model can run inference on real data."""
    model = build_test_binned_specialist_model()
    norm_quantizer = FixedPointQuantizer(integer_bits=1, fractional_bits=7)
    nn_quantizer = FixedPointQuantizer(integer_bits=2, fractional_bits=6)

    quantized_model = model.fixed_point_quantization(
        norm_quantizer=norm_quantizer,
        nn_quantizer=nn_quantizer,
    )

    data = make_dummy_data()
    result = quantized_model.predict(data, passthrough=False)
    assert isinstance(result, pl.DataFrame)
    assert "output" in result.columns
    assert "prediction" in result.columns
    assert result.height == N_SAMPLES


def test_binned_specialist_committee_fixed_point_quantization():
    m1 = build_test_binned_specialist_model()
    m2 = build_test_binned_specialist_model()
    committee = BinnedSpecialistCommittee(models=[m1, m2])

    norm_quantizer = FixedPointQuantizer(integer_bits=1, fractional_bits=7)
    nn_quantizer = FixedPointQuantizer(integer_bits=2, fractional_bits=6)

    quantized_committee = committee.fixed_point_quantization(
        norm_quantizer=norm_quantizer,
        nn_quantizer=nn_quantizer,
    )

    assert isinstance(quantized_committee, BinnedSpecialistCommittee)
    assert len(quantized_committee.models) == 2

    from neuralnet.normalizers.polars import FixedPointQuantizedAlternativeNorm1

    for m in quantized_committee.models:
        dense_layers = [l for l in m.keras_model.layers if isinstance(l, FixedPointQuantizationDense)]
        assert len(dense_layers) == 2
        assert isinstance(m.preprocessing.normalizer, FixedPointQuantizedAlternativeNorm1)


def test_training_job_with_hgq2_mlp(tmp_path: Path, ringer_dataset_dir: Path, isolated_executor):
    """Verify RingerKerasTrainingJob trains and reloads an HGQ2 quantized MLP with MinMax ranges."""
    future = isolated_executor.submit(
        _training_job_hgq2_routine,
        tmp_path=tmp_path,
        dataset_dir=ringer_dataset_dir,
    )
    future.result()


def _training_job_hgq2_routine(tmp_path: Path, dataset_dir: Path):
    os.environ["KERAS_BACKEND"] = "tensorflow"

    from hgq.layers import QDense

    from neuralnet.submitit import ExecutorConfig
    from neuralnet.workflows.ringer.training import RingerKerasTrainingJob

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    output_dir = tmp_path / "hgq2_output"

    job = RingerKerasTrainingJob(
        batch_size=32,
        data_table="data",
        dataset_dir=dataset_dir,
        et_col="et",
        et_bins=[{"low": 0.0, "high": 100000.0, "closed": "left"}],
        eta_col="eta",
        eta_bins=[{"low": 0.0, "high": 2.5, "closed": "left"}],
        fold_col="fold",
        kfold_table="kfold",
        label_col="label",
        norm_strategy="l1",
        rings_col="rings",
        ring_fraction=2,
        model_factory={
            "object_type": "mlp",
            "name": "hgq2_mlp",
            "layers": [
                {
                    "object_type": "hgq2_dense",
                    "units": 4,
                    "activation": "relu",
                    "name": "hidden_hgq",
                    "quantization": {
                        "object_type": "hgq2_fixed_point",
                        "i0": 2,
                        "f0": 6,
                        "ic": {"object_type": "min_max", "min_value": 0, "max_value": 4},
                        "fc": {"object_type": "min_max", "min_value": 2, "max_value": 10},
                    },
                },
                {
                    "object_type": "hgq2_dense",
                    "units": 1,
                    "activation": "sigmoid",
                    "name": "output_hgq",
                    "quantization": {
                        "object_type": "hgq2_fixed_point",
                        "i0": 2,
                        "f0": 6,
                    },
                },
            ],
        },
        loss={"object_type": "binary_cross_entropy", "from_logits": False},
        optimizer={"object_type": "adam", "learning_rate": 0.01},
        from_logits=False,
        epochs=1,
        inits=1,
        output_path=output_dir,
        executor_config=ExecutorConfig(
            cpus_per_task=1,
            executor_type="debug",
            logs_dir=logs_dir,
            name="test_hgq2_job",
        ),
        best_init={"key": "fit.val.max_sp", "criterion": "max"},
        best_fold={"key": "fit.val.max_sp", "criterion": "max"},
    )
    job.submit()

    loaded_job = RingerKerasTrainingJob.load(output_dir)
    member_id = loaded_job.selected_models.select(pl.col("id")).to_series().to_list()[0]
    reloaded_model = loaded_job.get_member_model(member_id)
    assert all(isinstance(layer, QDense) for layer in reloaded_model.layers)
