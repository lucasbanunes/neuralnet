import json
from pathlib import Path
from typing import Annotated

import pytest
import typer
from pydantic import Field, ValidationError
from typer.testing import CliRunner

from neuralnet.workflows.base import BaseActivity, register_activity_command


class DummyActivity(BaseActivity):
    value: Annotated[int, Field(gt=0, description="Positive integer parameter.")] = 1

    def submit(self):
        self.dump_config()
        return self.value


def test_base_activity_from_yaml_dump_config_and_reject_unknown(tmp_path: Path):
    """Verify BaseActivity.from_yaml, dump_config, and extra='forbid' rejection."""
    out_dir = tmp_path / "activity_out"
    yaml_path = tmp_path / "valid.yaml"
    yaml_path.write_text(f"output_path: {out_dir}\nvalue: 42\n", encoding="utf-8")

    activity = DummyActivity.from_yaml(yaml_path)
    assert activity.value == 42
    assert activity.output_path == out_dir

    result = activity.submit()
    assert result == 42
    assert (out_dir / "config.json").exists()
    dumped = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    assert dumped["value"] == 42
    assert dumped["output_path"] == str(out_dir)

    # Unknown YAML key must be rejected
    bad_yaml = tmp_path / "invalid.yaml"
    bad_yaml.write_text(f"output_path: {out_dir}\nvalue: 1\nunknown_field: true\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        DummyActivity.from_yaml(bad_yaml)


def test_register_activity_command_help_and_execution(tmp_path: Path):
    """Verify register_activity_command renders schema help and executes submit()."""
    app = typer.Typer(rich_markup_mode="markdown")
    register_activity_command(
        app=app,
        name="dummy",
        activity_cls=DummyActivity,
        short_help="Run dummy activity",
        description="A dummy activity used for testing.",
    )
    runner = CliRunner()

    help_res = runner.invoke(app, ["--help"])
    assert help_res.exit_code == 0
    assert "value" in help_res.output
    assert "output_path" in help_res.output

    out_dir = tmp_path / "cli_out"
    yaml_path = tmp_path / "cli.yaml"
    yaml_path.write_text(f"output_path: {out_dir}\nvalue: 7\n", encoding="utf-8")

    run_res = runner.invoke(app, ["--config", str(yaml_path)])
    assert run_res.exit_code == 0
    assert (out_dir / "config.json").exists()
