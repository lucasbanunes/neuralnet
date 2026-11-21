"""Ringer Committee Typer CLI using uniform BaseActivity registration."""

import typer

from neuralnet.workflows.base import register_activity_command
from neuralnet.workflows.ringer.alternative_norm1 import AlternativeNorm1Analysis
from neuralnet.workflows.ringer.inference import RingerCommitteeInferenceJob
from neuralnet.workflows.ringer.threshold_fit import RingerCommitteeThresholdFitJob
from neuralnet.workflows.ringer.training import RingerKerasTrainingJob

app = typer.Typer(help="Ringer Committee command line interface", rich_markup_mode="markdown")

register_activity_command(
    app=app,
    name="training",
    activity_cls=RingerKerasTrainingJob,
    short_help="Run Ringer Committee Trigger Training jobs",
    description=(
        "This command runs the training jobs for the Ringer Committee Trigger. "
        "The configuration for the training jobs is provided through a YAML file."
    ),
)

register_activity_command(
    app=app,
    name="threshold-fit",
    activity_cls=RingerCommitteeThresholdFitJob,
    short_help="Run Ringer Committee threshold fit job",
    description=(
        "This command evaluates a trained Ringer Committee by running inference "
        "on a (potentially different) dataset and computing metrics at configurable "
        "reference operating points. The configuration is provided through a YAML file."
    ),
)

register_activity_command(
    app=app,
    name="inference",
    activity_cls=RingerCommitteeInferenceJob,
    short_help="Run Ringer Committee inference job",
    description=(
        "This command runs full-dataset inference with a fitted Ringer Committee "
        "and saves the results as a parquet table keyed by ``id`` for later joining "
        "with other tables. The configuration is provided through a YAML file."
    ),
)

register_activity_command(
    app=app,
    name="alternative-norm1",
    activity_cls=AlternativeNorm1Analysis,
    short_help="Run AlternativeNorm1 quantization analysis",
    description=(
        "This command analyzes the impact of fixed-point quantization on AlternativeNorm1 "
        "by computing MSE, MAE, MAPE, and KL divergence across configurable bit ranges."
    ),
)
