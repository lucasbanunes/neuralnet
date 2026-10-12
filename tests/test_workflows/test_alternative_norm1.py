import pytest
import numpy as np
import polars as pl
from pathlib import Path

from neuralnet.workflows.ringer.alternative_norm1 import (
    IntegerRange,
    compute_norm_metrics,
    AlternativeNorm1Analysis,
)
from neuralnet.pydantic import pydantic_to_markdown_schema


def test_integer_range():
    r = IntegerRange(start=1, stop=4)
    assert r.start == 1
    assert r.stop == 4
    assert r.as_list() == [1, 2, 3]
    assert list(r) == [1, 2, 3]
    assert r.as_range() == range(1, 4)


def test_compute_norm_metrics_perfect_match():
    rng = np.random.default_rng(42)
    n_samples = 50
    n_rings = 10
    cols_norm = [f"r_{i}_norm" for i in range(n_rings)]
    cols_quant = [f"r_{i}_quant" for i in range(n_rings)]

    y = rng.uniform(0.01, 1.0, size=(n_samples, n_rings))
    y = y / np.sum(y, axis=1, keepdims=True)

    df_data = {}
    for i, (cn, cq) in enumerate(zip(cols_norm, cols_quant)):
        df_data[cn] = y[:, i]
        df_data[cq] = y[:, i]

    df = pl.DataFrame(df_data)
    metrics_lf, diffs_lf = compute_norm_metrics(df, norm_cols=cols_norm, quant_cols=cols_quant)

    assert isinstance(metrics_lf, pl.LazyFrame)
    assert isinstance(diffs_lf, pl.LazyFrame)

    metrics_df = metrics_lf.collect()
    diffs_df = diffs_lf.collect()

    assert metrics_df.height == 1
    assert metrics_df["mse"][0] == 0.0
    assert metrics_df["rmse"][0] == 0.0
    assert metrics_df["mae"][0] == 0.0
    assert metrics_df["mape"][0] == 0.0
    assert metrics_df["max_error"][0] == 0.0

    assert isinstance(diffs_df, pl.DataFrame)
    assert diffs_df.height == n_samples
    assert "sample_mse" in diffs_df.columns
    assert "sample_mae" in diffs_df.columns
    assert "sample_mape" in diffs_df.columns
    assert f"{cols_norm[0]}.diff" in diffs_df.columns
    assert f"{cols_norm[0]}.abs_diff" in diffs_df.columns
    assert f"{cols_norm[0]}.sq_diff" in diffs_df.columns
    assert f"{cols_norm[0]}.rel_diff" in diffs_df.columns

    # Test that calling with LazyFrame also returns LazyFrames without early collection
    metrics_lf2, diffs_lf2 = compute_norm_metrics(df.lazy(), norm_cols=cols_norm, quant_cols=cols_quant)
    assert isinstance(metrics_lf2, pl.LazyFrame)
    assert isinstance(diffs_lf2, pl.LazyFrame)


def test_alternative_norm1_analysis_submit(tmp_path: Path):
    rng = np.random.default_rng(42)
    n_samples = 50
    n_rings = 100

    rings_data = [rng.uniform(0.0, 10.0, n_rings).astype(np.float32).tolist() for _ in range(n_samples)]
    data_df = pl.DataFrame(
        {
            "id": np.arange(n_samples, dtype=np.int64),
            "rings": rings_data,
            "et": rng.uniform(15.0, 50.0, n_samples).astype(np.float32),
            "eta": rng.uniform(0.0, 2.4, n_samples).astype(np.float32),
        }
    )

    data_path = tmp_path / "data.parquet"
    data_df.write_parquet(data_path)

    output_dir = tmp_path / "results_dir"

    analysis = AlternativeNorm1Analysis(
        dataset_dir=tmp_path,
        data_table="data.parquet",
        rings_col="rings",
        ring_fraction=2,
        integer_bits_range={"start": 1, "stop": 2},
        fractional_bits_range={"start": 6, "stop": 10},
        output_path=output_dir,
    )

    results_df = analysis.submit()

    assert isinstance(results_df, pl.DataFrame)
    assert results_df.height == 4  # 1 integer bit * 4 fractional bit settings (6, 7, 8, 9)

    # Check directory outputs
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "README.md").exists()
    readme_text = (output_dir / "README.md").read_text()
    assert "AlternativeNorm1 Fixed-Point Quantization Analysis" in readme_text
    assert "results.csv" in readme_text
    assert "quantized/" in readme_text
    quantized_dir = output_dir / "quantized"
    assert quantized_dir.exists()

    diff_files = list(quantized_dir.glob("quantized_ib_*_fb_*.parquet"))
    assert len(diff_files) == 4

    sample_parquet = pl.read_parquet(diff_files[0])
    assert sample_parquet.height == n_samples
    assert "id" in sample_parquet.columns

    # Verify normalized and quantized ring columns exist
    first_ring_col = "rings.0_alternative_norm1"
    quant_ring_col = "rings.0_quantized_alternative_norm1"
    assert first_ring_col in sample_parquet.columns
    assert quant_ring_col in sample_parquet.columns

    expected_cols = [
        "integer_bits",
        "fractional_bits",
        "mse",
        "rmse",
        "mae",
        "mape",
        "max_error",
    ]
    for col in expected_cols:
        assert col in results_df.columns

    # As fractional bits increase (6 -> 7 -> 8 -> 9), errors should decrease monotonically
    mses = results_df["mse"].to_list()
    assert mses[0] > mses[1] > mses[2] > mses[3]

    maes = results_df["mae"].to_list()
    assert maes[0] > maes[1] > maes[2] > maes[3]

    # Test load, results property, and get_quantized_df method
    loaded_analysis = AlternativeNorm1Analysis.load(output_dir)
    assert loaded_analysis.rings_col == "rings"
    assert loaded_analysis.ring_fraction == 2
    assert loaded_analysis.output_path == output_dir

    # Test results property returns LazyFrame matching results.csv
    results_lazy = loaded_analysis.results
    assert isinstance(results_lazy, pl.LazyFrame)
    results_collected = results_lazy.collect()
    assert results_collected.height == 4
    assert "mse" in results_collected.columns

    # Test get_quantized_df returns LazyFrame for specific (ib, fb)
    diff_lazy = loaded_analysis.get_quantized_df(integer_bits=1, fractional_bits=6)
    assert isinstance(diff_lazy, pl.LazyFrame)
    diff_collected = diff_lazy.collect()
    assert diff_collected.height == n_samples
    assert first_ring_col in diff_collected.columns
    assert quant_ring_col in diff_collected.columns


def test_alternative_norm1_analysis_with_kfold(tmp_path: Path):
    rng = np.random.default_rng(42)
    n_samples = 40
    n_rings = 100

    rings_data = [rng.uniform(0.0, 5.0, n_rings).astype(np.float32).tolist() for _ in range(n_samples)]
    data_df = pl.DataFrame(
        {
            "id": np.arange(n_samples, dtype=np.int64),
            "rings": rings_data,
            "et": rng.uniform(15.0, 50.0, n_samples).astype(np.float32),
            "eta": rng.uniform(0.0, 2.4, n_samples).astype(np.float32),
        }
    )
    kfold_df = pl.DataFrame(
        {
            "id": np.arange(n_samples, dtype=np.int64),
            "label": rng.choice([0, 1], size=n_samples),
            "kfold": rng.choice([0, 1, 2, 3], size=n_samples),
        }
    )

    data_df.write_parquet(tmp_path / "data.parquet")
    kfold_df.write_parquet(tmp_path / "kfold.parquet")

    output_dir = tmp_path / "analysis_output"

    from neuralnet.bins import Bin, AbsoluteBin

    analysis = AlternativeNorm1Analysis(
        dataset_dir=tmp_path,
        data_table="data.parquet",
        kfold_table="kfold.parquet",
        rings_col="rings",
        ring_fraction=2,
        et_bins=[Bin(low=0.0, high=100.0, closed="left")],
        eta_bins=[AbsoluteBin(low=0.0, high=3.0, closed="left")],
        integer_bits_range=IntegerRange(start=1, stop=3),
        fractional_bits_range=IntegerRange(start=4, stop=7),
        output_path=output_dir,
    )

    results_df = analysis.submit()
    assert isinstance(results_df, pl.DataFrame)
    assert results_df.height == 6  # 2 int bits (1, 2) * 3 frac bits (4, 5, 6)
    assert "et_bin" in results_df.columns
    assert "eta_bin" in results_df.columns
    assert (output_dir / "results.csv").exists()
    assert (output_dir / "config.json").exists()
    assert (output_dir / "quantized").exists()
    assert len(list((output_dir / "quantized").glob("*.parquet"))) == 6

    loaded = AlternativeNorm1Analysis.load(output_dir)
    assert loaded.kfold_table == "kfold.parquet"
    assert loaded.results.collect().height == 6
    assert loaded.get_quantized_df(1, 4).collect().height == analysis.get_data().collect().height


def test_alternative_norm1_markdown_schema():
    schema = pydantic_to_markdown_schema(AlternativeNorm1Analysis)
    assert "integer_bits_range" in schema
    assert "rings_col" in schema
    assert "start" in schema
    assert "stop" in schema
