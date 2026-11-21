"""Shared YAML-activity base contract and uniform Typer CLI registration."""

from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ConfigDict, Field, PlainSerializer

from neuralnet.logging import LoggerName
from neuralnet.pydantic import YamlBaseModel, pydantic_to_markdown_schema


class BaseActivity(YamlBaseModel, ABC):
    """Base class for reproducible YAML-configured CLI activities.

    Enforces strict schema validation (``extra="forbid"``, ``frozen=True``),
    standardizes ``output_path`` and ``logger_name``, provides ``from_yaml``
    (inherited from :class:`~neuralnet.pydantic.YamlBaseModel`) and
    ``dump_config()`` snapshot persistence, and defines the abstract
    ``submit()`` entrypoint.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_path: Annotated[
        Path,
        Field(description="Directory where activity outputs and configuration snapshot are written."),
        PlainSerializer(str, return_type=str),
    ]
    logger_name: LoggerName = None

    @cached_property
    def config_path(self) -> Path:
        """Path to the serialized configuration snapshot inside ``output_path``."""
        return self.output_path / "config.json"

    def dump_config(self) -> Path:
        """Create ``output_path`` if needed and write ``config.json`` snapshot.

        Returns
        -------
        Path
            Path to the written ``config.json`` file.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=4))
        return self.config_path

    @abstractmethod
    def submit(self):
        """Execute or submit the activity."""
        ...


def register_activity_command(
    app: typer.Typer,
    name: str,
    activity_cls: type[BaseActivity],
    short_help: str,
    description: str,
) -> None:
    """Register a :class:`BaseActivity` subclass as a Typer CLI subcommand.

    Parameters
    ----------
    app : typer.Typer
        Target Typer application.
    name : str
        Subcommand name (e.g. ``"training"``, ``"threshold-fit"``).
    activity_cls : type[BaseActivity]
        Activity class implementing ``from_yaml`` and ``submit``.
    short_help : str
        One-line summary shown in CLI listings and help headers.
    description : str
        Detailed command description rendered before the auto-generated
        Pydantic markdown schema.
    """
    help_text = (
        f"{short_help}\n\n"
        f"{description}\n\n"
        f"The YAML file should follow the schema below:\n\n"
        f"{pydantic_to_markdown_schema(activity_cls)}"
    )

    @app.command(
        name=name,
        short_help=short_help,
        help=help_text,
    )
    def _command(
        config: Annotated[
            Path,
            typer.Option(
                "--config",
                help=f"Path to the YAML configuration file for the {name} activity.",
            ),
        ],
    ):
        activity = activity_cls.from_yaml(config)
        activity.submit()
