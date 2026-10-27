"""cli.run_scene: a terrain/data failure becomes a logged error and exit code 1."""
import logging

import pytest

from scene_generation import cli


def test_runtime_error_exits_with_code_1(caplog):
    def failing_scene(*args, **kwargs):
        raise RuntimeError("DEM terrain generation failed: no tiles")

    logger = logging.getLogger("test_cli")
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        cli.run_scene(logger, failing_scene, "points", "data_dir")
    assert exc.value.code == 1
    assert "DEM terrain generation failed" in caplog.text


def test_successful_scene_passes_through():
    seen = {}

    def scene(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        return "building map"

    assert cli.run_scene(logging.getLogger("test_cli"), scene, 1, 2, flag=True) == "building map"
    assert seen == {"args": (1, 2), "kwargs": {"flag": True}}


def test_other_errors_are_not_swallowed():
    def bad_input(*args, **kwargs):
        raise ValueError("Invalid ground material type")

    with pytest.raises(ValueError):
        cli.run_scene(logging.getLogger("test_cli"), bad_input)
