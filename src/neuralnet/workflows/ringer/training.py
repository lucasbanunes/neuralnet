from collections import defaultdict

import polars as pl
from typing import (
    Annotated,
    Literal,
    Self,
    TypedDict,
    TYPE_CHECKING,
    overload,
)

from neuralnet import get_logger

if TYPE_CHECKING:
    from hgq.config import QuantizerConfig
    from keras import Sequential

from itertools import product
from pydantic import AfterValidator, Field, ConfigDict, BaseModel
import logging
from pathlib import Path
import numpy as np
from functools import cached_property
import tempfile
import time
import mlflow
import mlflow.keras
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

from ...submitit import ExecutorConfig
from ...logging import LoggerName
from ...datasets.ringer import (
    RingerParquetDataset,
    DataTableType,
    RingsColType,
    KFoldTableType,
    LabelColType,
    FoldColType,
)
from ...models.keras.factories import (
    EpochsType,
    VerboseType,
    FitRoutineDict,
    LossType,
    OptimizerType,
)
from ...models.binned_committee import VariableBin
from .models import BinnedSpecialistCommittee, BinnedSpecialistModel, NormStrategyType
from .fields import RingFractionType
from ...models.dense import (
    MLPFactory,
    FixedPointQuantizedMLP,
)
from ...utils import traverse, unflatten_mapping
from ...normalizers.factories import FixedPointAlternativeNormL1
from ...normalizers.polars import AlternativeNorm1, FixedPointQuantizedAlternativeNorm1
from ...utils.polars import RingSlicesPerLayer
from ...datasets import DirectoryType
from ...pydantic import YamlBaseModel
from ...bins import (
    Bin,
    BinDict,
    AbsoluteVariableBin,
    AbsoluteBin,
    AbsoluteBinDict,
)
from ...quantization.quantizers import FixedPointQuantizer
from ...quantization.keras import FixedPointQuantizationDense

type BalanceClassWeightsType = Annotated[
    bool,
    Field(
        description="Whether to balance class weights during training. If True, the class weights will be set inversely proportional to the class frequencies.",
    ),
]

type BatchSizeType = Annotated[
    int,
    Field(
        gt=0,
        description="Batch size for the dataset. Must be a positive integer.",
    ),
]

type EtBinType = Annotated[
    list[Bin],
    Field(
        description="Bins to be used for the Et variable.",
        min_length=1,
    ),
]

type EtaBinType = Annotated[
    list[AbsoluteBin],
    Field(
        description="Bins to be used for the Eta variable.",
        min_length=1,
    ),
]

type EtColType = Annotated[str, Field(description="Name of the et column in the data table")]

type EtaColType = Annotated[str, Field(description="Name of the eta column in the data table")]

type FromLogitsType = Annotated[
    bool,
    Field(
        description="Whether the model output to consider is logits. When enabled considers the threshold limits as probabilities between 0 and 1",
    ),
]

type ExecutorConfigType = Annotated[
    ExecutorConfig,
    Field(description="Slurm configuration for running the training job on a Slurm cluster"),
]


class HistoryTrainMetrics(TypedDict):
    loss: list[float]
    accuracy: list[float]


class HistoryValMetrics(TypedDict):
    loss: list[float]
    accuracy: list[float]
    max_sp: list[float]
    max_sp_pd: list[float]
    max_sp_fa: list[float]
    max_sp_threshold: list[float]
    max_sp_best_epoch: list[int]


class HistoryDict(FitRoutineDict):
    train: HistoryTrainMetrics
    val: HistoryValMetrics


class FitTrainMetrics(TypedDict):
    loss: float
    accuracy: float


class FitValMetrics(TypedDict):
    loss: float
    accuracy: float
    max_sp: float
    max_sp_pd: float
    max_sp_fa: float
    max_sp_threshold: float
    max_sp_best_epoch: int


class FitDict(TypedDict):
    train: FitTrainMetrics
    val: FitValMetrics


# class EvaluationDict(EnhancedConfusionMatrixDict):
#     loss: float
#     weighted: NotRequired[EnhancedConfusionMatrixDict]


class TrainingResult(TypedDict):
    fold: int
    init: int
    et_bin: BinDict
    eta_bin: AbsoluteBinDict
    history: HistoryDict
    fit: FitDict
    # train: EvaluationDict
    # val: EvaluationDict
    # test: EvaluationDict


type PatienceType = Annotated[
    int,
    Field(
        description="Number of epochs with no improvement after which training will be stopped.",
    ),
]


def validate_selection_criteria_key(value: str) -> str:
    valid_keys_list = [f"fit.train.{metric}" for metric in FitTrainMetrics.__annotations__] + [
        f"fit.val.{metric}" for metric in FitValMetrics.__annotations__
    ]
    if value not in valid_keys_list:
        raise ValueError(f"Invalid selection criteria key: {value}. Must be one of {valid_keys_list}")
    return value


type SelectionCriteriaKeyType = Annotated[
    str,
    Field(
        description="Criterion to select the best model initilization or fold groups. Must be a key in the fit dictionary of the training results.",
    ),
    AfterValidator(validate_selection_criteria_key),
]


class SelectionCriteria(BaseModel):
    key: SelectionCriteriaKeyType
    criterion: Literal["max", "min"] = Field(
        "max",
        description="How to select the best model from initilization or fold groups. Must be either 'max' or 'min'.",
    )

    def select_best_expr(self) -> pl.Expr:
        if self.criterion == "max":
            return pl.all().sort_by(self.key, descending=True).first()
        elif self.criterion == "min":
            return pl.all().sort_by(self.key, descending=False).first()
        else:
            raise ValueError(f"Invalid criterion: {self.criterion}. Must be either 'max' or 'min'.")


type EtaBinIntervalValue = Annotated[
    float,
    Field(
        ge=0,
        description="Represents the value of the Eta bin interval. Must be a non-negative integer.",
    ),
]


class BinValidator:
    def __init__(self, bin_class: type[Bin]):
        self.bin_class = bin_class

    def __call__(self, et_bins: list[Bin | float]) -> list[Bin]:
        length = len(et_bins)
        is_float = isinstance(et_bins[0], float)
        if length < 2 and is_float:
            raise ValueError(f"et_bins must have at least 2 values to define a bin. Got {length} value(s).")
        if is_float:
            et_bins = [self.bin_class(low=et_bins[i], high=et_bins[i + 1], closed="left") for i in range(length - 1)]
            return et_bins
        return et_bins


type QuantizerConfigType = "QuantizerConfig" | None


class PreprocessingPipeline:
    def __init__(
        self,
        ring_selector: RingSlicesPerLayer,
        normalizer: AlternativeNorm1 | FixedPointQuantizedAlternativeNorm1 | None = None,
    ):
        """Create a preprocessing pipeline for ring features.

        Parameters
        ----------
        ring_selector : RingSlicesPerLayer
            Ring selection and expansion transform.
        normalizer : AlternativeNorm1 or FixedPointQuantizedAlternativeNorm1 or None, default=None
            Optional normalization stage applied after ring selection.
        """

        self.ring_selector = ring_selector
        self.normalizer = normalizer

    @classmethod
    def from_job_params(
        cls,
        rings_col: RingsColType,
        ring_fraction: RingFractionType = 2,
        norm_strategy: NormStrategyType = None,
    ) -> "PreprocessingPipeline":
        """Build a preprocessing pipeline from training job parameters.

        Parameters
        ----------
        rings_col : RingsColType
            Input rings column name.
        ring_fraction : RingFractionType, default=2
            Fraction of rings retained per layer.
        norm_strategy : {"l1"} or None, default=None
            Optional normalization strategy.

        Returns
        -------
        PreprocessingPipeline
            Configured preprocessing pipeline.

        Raises
        ------
        ValueError
            If ``norm_strategy`` is not supported.
        """

        ring_selector = RingSlicesPerLayer(
            rings_col=rings_col,
            fraction=ring_fraction,
            output_format="expanded_columns",
        )

        if norm_strategy == "l1":
            normalizer = AlternativeNorm1(
                input_cols=ring_selector.output_cols,
            )
        elif isinstance(norm_strategy, FixedPointAlternativeNormL1):
            normalizer = norm_strategy.as_polars_transform(input_cols=ring_selector.output_cols)
        elif norm_strategy is None:
            normalizer = None
        else:
            raise ValueError(f"Unsupported norm_strategy: {norm_strategy}")

        return cls(
            ring_selector=ring_selector,
            normalizer=normalizer,
        )

    @property
    def input_cols(self) -> list[str]:
        """Columns required by the preprocessing pipeline.

        Returns
        -------
        list[str]
            Input column names required by ring selection.
        """

        return self.ring_selector.input_cols

    @property
    def output_cols(self) -> list[str]:
        """Columns produced by the preprocessing pipeline.

        Returns
        -------
        list[str]
            Output feature columns consumed by downstream models.
        """

        if self.normalizer is None:
            return self.ring_selector.output_cols

        return self.normalizer.output_cols

    @cached_property
    def aux_cols(self) -> list[str]:
        """Auxiliary columns produced by intermediate preprocessing stages.

        Returns
        -------
        list[str]
            Intermediate ring-selector columns retained when normalization is
            active; empty when no normalizer is configured.
        """

        if self.normalizer is None:
            return []
        return self.ring_selector.output_cols

    def __call__(self, data: pl.DataFrame | pl.LazyFrame, passthrough: bool = False) -> pl.DataFrame | pl.LazyFrame:
        """Apply preprocessing pipeline transformations.

        Parameters
        ----------
        data : pl.DataFrame or pl.LazyFrame
            Input table to preprocess.
        passthrough : bool, default=False
            Whether to preserve non-feature columns in intermediate transforms.

        Returns
        -------
        pl.DataFrame or pl.LazyFrame
            Preprocessed table.
        """

        data = data.pipe(self.ring_selector, passthrough=passthrough)
        if self.normalizer:
            data = data.pipe(self.normalizer, passthrough=passthrough)
        return data

    def fixed_point_quantization(self, norm_quantizer: FixedPointQuantizer) -> Self:
        """Create a fixed-point quantized variant of the pipeline.

        Parameters
        ----------
        norm_quantizer : FixedPointQuantizer
            Quantizer applied to the normalization stage when available.

        Returns
        -------
        Self
            Quantized preprocessing pipeline.
        """

        if self.normalizer is None:
            return self

        if self.normalizer is None:
            logger = get_logger()
            logger.warning("Normalization strategy is None. Fixed Point quantization will not change the behavior.")
            quantized_normalizer = None
        else:
            quantized_normalizer = self.normalizer.fixed_point_quantization(norm_quantizer)

        return PreprocessingPipeline(self.ring_selector, quantized_normalizer)


class BinDict(TypedDict):
    low: float
    high: float
    closed: Literal["left", "right", "both", "neither"]


class RingerKerasTrainingJob(YamlBaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Dataset params
    batch_size: BatchSizeType = 32
    data_table: DataTableType
    dataset_dir: DirectoryType
    et_col: EtColType
    et_bins: EtBinType
    eta_col: EtaColType
    eta_bins: EtaBinType
    fold_col: FoldColType
    kfold_table: KFoldTableType
    label_col: LabelColType
    norm_strategy: NormStrategyType = None
    rings_col: RingsColType
    ring_fraction: RingFractionType = 2

    # Model params
    model_factory: Annotated[
        MLPFactory | FixedPointQuantizedMLP,
        Field(
            ...,
            discriminator="object_type",
            description="Description of the MLP model to train.",
        ),
    ]
    from_logits: FromLogitsType = False
    epochs: EpochsType = 5000
    verbose: VerboseType = 1
    loss: LossType
    optimizer: OptimizerType
    balance_class_weights: BalanceClassWeightsType = True
    inits: Annotated[int, Field(description="Number of initializations")] = 5
    patience: PatienceType = 25

    # Threshold Fit
    # num_thresholds: int = Field(
    #     ge=2,
    #     description="Number of thresholds to evaluate for the strategy.",
    # )
    # lower_threshold: float = Field(
    #     ge=0,
    #     lt=1,
    #     description="Lower threshold limit for the strategy.",
    # )
    # upper_threshold: float = Field(
    #     gt=0,
    #     le=1,
    #     description="Upper threshold limit for the strategy.",
    # )

    # Execution related fields
    dry_run: Annotated[bool, Field(description="Perform a dry run without actually training")] = False
    executor_config: ExecutorConfigType

    # MLflow configuration fields
    experiment_name: Annotated[
        str,
        Field(description="Name of the MLflow experiment to log the training runs under."),
    ]
    tracking_uri: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "MLflow tracking server URI. If None, uses the MLFLOW_TRACKING_URI "
                "environment variable or default local MLflow store."
            ),
        ),
    ] = None
    run_name: Annotated[
        str | None,
        Field(
            default=None,
            description="Optional name for the main (parent) MLflow run.",
        ),
    ] = None
    run_id: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "MLflow run ID of the main training run. Automatically populated "
                "when the job is submitted or loaded from MLflow."
            ),
        ),
    ] = None

    # Misc
    logger_name: LoggerName = None

    # Model Selection
    best_init: SelectionCriteria = Field(
        default_factory=SelectionCriteria,
        description="Criterion to select the best model initialization. Must be a key in the fit dictionary of the training results.",
    )
    best_fold: SelectionCriteria = Field(
        default_factory=SelectionCriteria,
        description="Criterion to select the best fold. Must be a key in the fit dictionary of the training results.",
    )

    def _configure_mlflow(self) -> MlflowClient:
        if self.tracking_uri is not None:
            mlflow.set_tracking_uri(self.tracking_uri)
        return MlflowClient(tracking_uri=self.tracking_uri)

    @cached_property
    def preprocessing_pipeline(self) -> PreprocessingPipeline:
        return PreprocessingPipeline.from_job_params(
            ring_fraction=self.ring_fraction,
            rings_col=self.rings_col,
            norm_strategy=self.norm_strategy,
        )

    @cached_property
    def all_model_results(self) -> pl.DataFrame:
        if self.run_id is None:
            raise ValueError("Cannot load all_model_results because run_id is not set.")
        self._configure_mlflow()
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id,
                artifact_path="all_models_results.parquet",
                dst_path=tmp_dir,
                tracking_uri=self.tracking_uri,
            )
            return pl.read_parquet(local_path)

    @cached_property
    def selected_models(self) -> pl.DataFrame:
        if self.run_id is None:
            raise ValueError("Cannot load selected_models because run_id is not set.")
        self._configure_mlflow()
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id,
                artifact_path="selected_models.parquet",
                dst_path=tmp_dir,
                tracking_uri=self.tracking_uri,
            )
            return pl.read_parquet(local_path)

    def get_dataset(
        self, fold: int, et_bin: BinDict | None = None, eta_bin: BinDict | None = None, **kwargs
    ) -> RingerParquetDataset:
        """Instantiate a dataset view for a specific fold and bin pair.

        Parameters
        ----------
        fold : int
            Fold index.
        et_bin : BinDict | None, default=None
            Et bin definition.
        eta_bin : BinDict | None, default=None
            Eta bin definition.
        **kwargs
            Additional keyword arguments forwarded to ``RingerParquetDataset``.

        Returns
        -------
        RingerParquetDataset
            Configured dataset instance.
        """
        if et_bin is not None:
            et_bin = {"var_name": self.et_col, **et_bin}

        if eta_bin is not None:
            eta_bin = {"var_name": self.eta_col, **eta_bin}

        config_dict = {
            "dataset_dir": self.dataset_dir,
            "data_table": self.data_table,
            "rings_col": self.rings_col,
            "kfold_table": self.kfold_table,
            "label_col": self.label_col,
            "fold_col": self.fold_col,
            "fold": fold,
            "et_bin": et_bin,
            "eta_bin": eta_bin,
        }
        config_dict.update(kwargs)
        dataset = RingerParquetDataset(**config_dict)
        return dataset

    def get_numpy_data(self, df: pl.LazyFrame) -> tuple[np.ndarray, np.ndarray]:
        """Materialize model-ready NumPy arrays from a lazy dataset frame.

        Parameters
        ----------
        df : pl.LazyFrame
            Input dataset split.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Tuple ``(X, y)`` with preprocessed feature matrix and labels.
        """

        df = df.pipe(self.preprocessing_pipeline, passthrough=True)
        X = df.select(self.preprocessing_pipeline.output_cols)
        y = df.select(self.label_col)
        X, y = pl.collect_all([X, y])
        X = X.to_numpy()
        y = y.to_numpy().flatten()
        return X, y

    def get_specialist_model(
        self,
        member_id: int,
        et_col: str | None = None,
        eta_col: str | None = None,
        rings_col: str | None = None,
    ) -> BinnedSpecialistModel:
        """Build one specialist model from a selected trained member.

        Parameters
        ----------
        member_id : int
            Identifier of the selected member.
        et_col : str or None, default=None
            Et column name. Uses job default when ``None``.
        eta_col : str or None, default=None
            Eta column name. Uses job default when ``None``.
        rings_col : str or None, default=None
            Rings column name. Uses job default when ``None``.

        Returns
        -------
        BinnedSpecialistModel
            Specialist model bound to the selected member bins and metadata.

        Raises
        ------
        ValueError
            If no or multiple model rows match ``member_id``.
        """

        if et_col is None:
            et_col = self.et_col

        if eta_col is None:
            eta_col = self.eta_col

        if rings_col is None:
            rings_col = self.rings_col

        preprocessing_pipeline = PreprocessingPipeline.from_job_params(
            rings_col=rings_col,
            ring_fraction=self.ring_fraction,
            norm_strategy=self.norm_strategy,
        )

        member_rows = self.all_model_results.filter(pl.col("id") == member_id)
        if member_rows.height == 0:
            raise ValueError(f"No selected model entry found for member_id={member_id}.")
        if member_rows.height > 1:
            raise ValueError(f"Multiple model entries found for member_id={member_id}.")
        row = member_rows.row(0, named=True)

        keras_model, training_results = self.get_member_model(member_id, with_results=True)
        training_results["id"] = member_id

        return BinnedSpecialistModel(
            bins=[
                VariableBin(
                    var_name=et_col,
                    low=row["et_bin.low"],
                    high=row["et_bin.high"],
                    closed=row["et_bin.closed"],
                ),
                AbsoluteVariableBin(
                    var_name=eta_col,
                    low=row["eta_bin.low"],
                    high=row["eta_bin.high"],
                    closed=row["eta_bin.closed"],
                ),
            ],
            keras_model=keras_model,
            preprocessing=preprocessing_pipeline,
            training_results=training_results,
        )

    def get_specialist_committee(
        self,
        et_col: str | None = None,
        eta_col: str | None = None,
        rings_col: str | None = None,
    ) -> BinnedSpecialistCommittee:
        """Build a specialist committee from selected trained members.

        Parameters
        ----------
        et_col : str or None, default=None
            Et column name. Uses job default when ``None``.
        eta_col : str or None, default=None
            Eta column name. Uses job default when ``None``.
        rings_col : str or None, default=None
            Rings column name. Uses job default when ``None``.

        Returns
        -------
        BinnedSpecialistCommittee
            Committee composed of all selected specialist members.
        """

        if et_col is None:
            et_col = self.et_col

        if eta_col is None:
            eta_col = self.eta_col

        if rings_col is None:
            rings_col = self.rings_col

        member_ids = self.selected_models.select(pl.col("id")).to_series().to_list()
        binned_models = [
            self.get_specialist_model(
                member_id=member_id,
                et_col=et_col,
                eta_col=eta_col,
                rings_col=rings_col,
            )
            for member_id in member_ids
        ]

        return BinnedSpecialistCommittee(models=binned_models)

    def submit(self):
        logger = logging.getLogger(self.logger_name)
        logger.info(f"Starting job with model {self.model_factory.name} on dataset at {self.dataset_dir}")

        client = self._configure_mlflow()
        experiment = mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(
            run_id=self.run_id,
            experiment_id=experiment.experiment_id if self.run_id is None else None,
            run_name=self.run_name if self.run_id is None else None,
        ) as parent_run:
            parent_run_id = parent_run.info.run_id
            object.__setattr__(self, "run_id", parent_run_id)
            config_dict = self.model_dump(exclude={"run_id"})
            flat_params = {k: str(v) for k, v in traverse(config_dict, include_sequences=True)}
            mlflow.log_params(flat_params)

        dataset = RingerParquetDataset(
            dataset_dir=self.dataset_dir,
            data_table=self.data_table,
            rings_col=self.rings_col,
            kfold_table=self.kfold_table,
            label_col=self.label_col,
            fold_col=self.fold_col,
            fold=0,
        )

        n_folds = dataset.get_n_folds()
        logger.info(f"The dataset has {n_folds} folds.")

        existing_runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}' and attributes.status = 'FINISHED'",
        )
        completed_member_ids = {int(r.data.params["id"]) for r in existing_runs if "id" in r.data.params}

        folds_range = range(n_folds)
        inits_range = range(self.inits)
        iterator = product(self.et_bins, self.eta_bins, folds_range, inits_range)
        executor = self.executor_config.get_executor()
        submitted_jobs = []
        with executor.batch():
            for member_id, (et_bin, eta_bin, fold, init) in enumerate(iterator):
                if member_id in completed_member_ids:
                    logger.info(
                        f"{member_id}: Nested MLflow run for member {member_id} already finished. Skipping training."
                    )
                    continue
                logger.info(
                    f"{member_id}: Submitting training for Et bin {et_bin} and Eta bin {eta_bin}, fold {fold} and init {init}"
                )
                submitted_job = executor.submit(
                    self._run_training,
                    member_id,
                    et_bin.model_dump(),
                    eta_bin.model_dump(),
                    fold,
                    init,
                    parent_run_id,
                    experiment.experiment_id,
                )
                submitted_jobs.append(submitted_job)

        dependent_executor = self.executor_config.get_executor()
        from submitit import AutoExecutor, LocalExecutor

        if isinstance(dependent_executor, AutoExecutor) and submitted_jobs:
            dependency_string = ":".join(str(job.job_id) for job in submitted_jobs)
            logger.info(f"Submitting dependent job with dependency on jobs: {dependency_string}")
            dependent_executor.update_parameters(
                slurm_additional_parameters={"dependency": f"afterok:{dependency_string}"}
            )
            dependent_executor.submit(self.post_training, parent_run_id, experiment.experiment_id)
        elif isinstance(dependent_executor, LocalExecutor):
            for job in submitted_jobs:
                job.result()
            post_job = dependent_executor.submit(self.post_training, parent_run_id, experiment.experiment_id)
            post_job.result()
        else:
            dependent_executor.submit(self.post_training, parent_run_id, experiment.experiment_id)

        logger.info("All jobs submitted.")

    def _run_training(
        self,
        member_id: int,
        et_bin: dict,
        eta_bin: dict,
        fold: int,
        init: int,
        parent_run_id: str | None = None,
        experiment_id: str | None = None,
    ):
        from neuralnet.models.keras.routines import fit_routine
        from keras import Model

        logger = logging.getLogger(self.logger_name)
        logger.info(
            f"{member_id}: Running training for Et bin {et_bin} and Eta bin {eta_bin}, fold {fold} and init {init}"
        )

        client = self._configure_mlflow()
        resolved_parent_run_id = parent_run_id or self.run_id
        if experiment_id is None:
            experiment = mlflow.set_experiment(self.experiment_name)
            experiment_id = experiment.experiment_id

        dataset = self.get_dataset(
            fold=fold,
            et_bin=et_bin,
            eta_bin=eta_bin,
        )

        results: TrainingResult = {
            "fold": fold,
            "init": init,
            "et_bin": et_bin,
            "eta_bin": eta_bin,
        }
        model_factory = self.model_factory.model_copy(update={"name": f"{self.model_factory.name}_{member_id}"})
        model: Model = model_factory.as_keras()
        if self.balance_class_weights:
            class_weights = dataset.train_class_weights()
        else:
            class_weights = None

        if self.dry_run:
            logger.warning("Dry run enabled, running one epoch.")
            epochs = 1
        else:
            epochs = self.epochs

        from neuralnet.callbacks.keras import SP

        train_data = self.get_numpy_data(dataset.train_df())
        val_data = self.get_numpy_data(dataset.val_df())

        sp_callback = SP(
            validation_data=val_data,
            patience=self.patience,
            verbose=self.verbose,
            save_the_best=True,
            from_logits=self.from_logits,
        )
        callbacks = [sp_callback]

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=f"member_{member_id}",
            nested=True,
            parent_run_id=resolved_parent_run_id,
        ) as child_run:
            member_params = {
                "id": member_id,
                "fold": fold,
                "init": init,
                "et_bin": et_bin,
                "eta_bin": eta_bin,
            }
            flat_member_params = {k: str(v) for k, v in traverse(member_params, include_sequences=True)}
            mlflow.log_params(flat_member_params)

            model, results["history"] = fit_routine(
                model=model,
                train_data=train_data,
                val_data=val_data,
                loss=self.loss.as_keras(),
                optimizer=self.optimizer.as_keras(),
                metrics=["accuracy"],
                callbacks=callbacks,
                epochs=epochs,
                verbose=self.verbose,
                batch_size=self.batch_size,
                class_weight=class_weights,
            )
            logger.info(f"Finished training for fold {fold} and init {init}")

            start_dt = results["history"]["start"]
            end_dt = results["history"]["end"]
            training_time = (end_dt - start_dt).total_seconds()
            mlflow.set_tags(
                {
                    "history.start": start_dt.isoformat(),
                    "history.end": end_dt.isoformat(),
                }
            )

            best_epoch = results["history"]["val"]["max_sp_best_epoch"][-1]
            results["fit"] = {
                "train": {
                    metric_name: metric_history[best_epoch]
                    for metric_name, metric_history in results["history"]["train"].items()
                },
                "val": {
                    metric_name: metric_history[best_epoch]
                    for metric_name, metric_history in results["history"]["val"].items()
                },
            }

            logger.info(f"Finished evaluating for fold {fold} and init {init}")

            ts = int(time.time() * 1000)
            metric_entities: list[Metric] = []
            for split in ("train", "val"):
                for metric_name, metric_history in results["history"][split].items():
                    key = f"history.{split}.{metric_name}"
                    for step, val in enumerate(metric_history):
                        metric_entities.append(Metric(key=key, value=float(val), timestamp=ts, step=step))

            for key, val in traverse({"fit": results["fit"]}, include_sequences=False):
                metric_entities.append(Metric(key=key, value=float(val), timestamp=ts, step=int(best_epoch)))

            metric_entities.append(Metric(key="training_time", value=float(training_time), timestamp=ts, step=0))

            for idx in range(0, len(metric_entities), 1000):
                client.log_batch(child_run.info.run_id, metrics=metric_entities[idx : idx + 1000])

            mlflow.keras.log_model(model, name="model", pip_requirements=[])
            logger.info(f"Logged model and metrics to MLflow child run {child_run.info.run_id}")

        del model
        from keras.backend import clear_session

        clear_session()

    def post_training(self, parent_run_id: str | None = None, experiment_id: str | None = None):
        logger = logging.getLogger(self.logger_name)
        client = self._configure_mlflow()

        resolved_parent_run_id = parent_run_id or self.run_id
        if resolved_parent_run_id is None:
            raise ValueError("parent_run_id must be provided or set on self.run_id before running post_training.")

        if experiment_id is None:
            parent_run_info = client.get_run(resolved_parent_run_id)
            experiment_id = parent_run_info.info.experiment_id

        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{resolved_parent_run_id}'",
        )

        all_models_results = defaultdict(list)
        for child_run in child_runs:
            params = child_run.data.params
            metrics = child_run.data.metrics
            tags = child_run.data.tags
            c_run_id = child_run.info.run_id

            member_id = int(params["id"])
            all_models_results["id"].append(member_id)
            all_models_results["run_id"].append(c_run_id)
            all_models_results["fold"].append(int(params["fold"]))
            all_models_results["init"].append(int(params["init"]))
            all_models_results["et_bin.low"].append(float(params["et_bin.low"]))
            all_models_results["et_bin.high"].append(float(params["et_bin.high"]))
            all_models_results["et_bin.closed"].append(params["et_bin.closed"])
            all_models_results["eta_bin.low"].append(float(params["eta_bin.low"]))
            all_models_results["eta_bin.high"].append(float(params["eta_bin.high"]))
            all_models_results["eta_bin.closed"].append(params["eta_bin.closed"])
            all_models_results["history.start"].append(tags.get("history.start"))
            all_models_results["history.end"].append(tags.get("history.end"))
            all_models_results["training_time"].append(float(metrics["training_time"]))

            for metric_name in FitTrainMetrics.__annotations__:
                fit_key = f"fit.train.{metric_name}"
                all_models_results[fit_key].append(float(metrics[fit_key]))

                hist_key = f"history.train.{metric_name}"
                hist_items = sorted(client.get_metric_history(c_run_id, hist_key), key=lambda m: m.step)
                all_models_results[hist_key].append([float(m.value) for m in hist_items])

            for metric_name in FitValMetrics.__annotations__:
                fit_key = f"fit.val.{metric_name}"
                is_int_metric = metric_name == "max_sp_best_epoch"
                val = int(metrics[fit_key]) if is_int_metric else float(metrics[fit_key])
                all_models_results[fit_key].append(val)

                hist_key = f"history.val.{metric_name}"
                hist_items = sorted(client.get_metric_history(c_run_id, hist_key), key=lambda m: m.step)
                all_models_results[hist_key].append(
                    [int(m.value) if is_int_metric else float(m.value) for m in hist_items]
                )

        logger.info(
            f"Computing best models based on the selection criteria: init: {self.best_init}, fold: {self.best_fold}"
        )
        all_model_results_df = (
            pl.DataFrame(all_models_results)
            .with_columns(
                pl.col("history.start").str.to_datetime(),
                pl.col("history.end").str.to_datetime(),
            )
            .sort("id")
        )

        bins_cols = [
            "et_bin.low",
            "et_bin.high",
            "et_bin.closed",
            "eta_bin.low",
            "eta_bin.high",
            "eta_bin.closed",
        ]
        best_init_results = all_model_results_df.group_by(*bins_cols, "fold").agg(self.best_init.select_best_expr())
        selected_models = best_init_results.group_by(*bins_cols).agg(self.best_fold.select_best_expr()).sort("id")

        with mlflow.start_run(run_id=resolved_parent_run_id):
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir_path = Path(tmp_dir)
                all_results_file = tmp_dir_path / "all_models_results.parquet"
                selected_models_file = tmp_dir_path / "selected_models.parquet"
                all_model_results_df.write_parquet(all_results_file)
                selected_models.write_parquet(selected_models_file)
                mlflow.log_artifact(str(all_results_file))
                mlflow.log_artifact(str(selected_models_file))

            parent_metrics: dict[str, float] = {}
            for row in selected_models.iter_rows(named=True):
                member_id = int(row["id"])
                child_run_id = row["run_id"]
                parent_metrics[f"selected.member_{member_id}.training_time"] = float(row["training_time"])
                for metric_name in FitTrainMetrics.__annotations__:
                    parent_metrics[f"selected.member_{member_id}.fit.train.{metric_name}"] = float(
                        row[f"fit.train.{metric_name}"]
                    )
                for metric_name in FitValMetrics.__annotations__:
                    parent_metrics[f"selected.member_{member_id}.fit.val.{metric_name}"] = float(
                        row[f"fit.val.{metric_name}"]
                    )

                keras_model = mlflow.keras.load_model(
                    f"runs:/{child_run_id}/model",
                    custom_objects={"FixedPointQuantizationDense": FixedPointQuantizationDense},
                )
                mlflow.keras.log_model(keras_model, name=f"model_member_{member_id}", pip_requirements=[])

            if parent_metrics:
                mlflow.log_metrics(parent_metrics)

    @classmethod
    def load(cls, run_id: str, tracking_uri: str | None = None) -> Self:
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        run = client.get_run(run_id)
        config = unflatten_mapping(run.data.params, parse_sequences=True, parse_none=True)
        config["run_id"] = run_id
        if tracking_uri is not None:
            config["tracking_uri"] = tracking_uri
        return cls(**config)

    @overload
    def get_member_model(self, member_id: int, with_results: Literal[False] = False) -> "Sequential": ...

    @overload
    def get_member_model(self, member_id: int, with_results: Literal[True]) -> tuple["Sequential", TrainingResult]: ...

    def get_member_model(self, member_id: int, with_results: bool = False):
        self._configure_mlflow()
        member_rows = self.all_model_results.filter(pl.col("id") == member_id)
        if member_rows.height == 0:
            raise ValueError(f"No model entry found for member_id={member_id}.")
        row = member_rows.row(0, named=True)
        child_run_id = row["run_id"]

        model = mlflow.keras.load_model(
            f"runs:/{child_run_id}/model",
            custom_objects={"FixedPointQuantizationDense": FixedPointQuantizationDense},
        )

        if not with_results:
            return model

        serialized_row = {
            k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items() if k not in ("id", "run_id")
        }
        results = unflatten_mapping(serialized_row)
        return model, results
