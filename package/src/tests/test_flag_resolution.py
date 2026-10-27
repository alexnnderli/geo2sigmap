"""
Main bugs 1 and 2: which terrain source is used, and when roads are dropped.

Before the fix, --enable-dem-terrain only did anything when LiDAR terrain was
also on, and --generate-roads was silently ignored whenever it couldn't run.
"""
import logging

import pytest

from scene_generation import core


@pytest.mark.parametrize(
    "lidar, dem, expected",
    [
        (False, False, None),
        (True, False, core.TERRAIN_MODE_LIDAR),
        (False, True, core.TERRAIN_MODE_DEM),  # DEM works on its own
        (True, True, core.TERRAIN_MODE_DEM),   # DEM wins when both are set
    ],
)
def test_resolve_terrain_mode(lidar, dem, expected):
    assert core.resolve_terrain_mode(lidar, dem) == expected


def test_both_terrain_flags_warns_that_lidar_terrain_is_ignored(caplog):
    with caplog.at_level(logging.WARNING, logger=core.logger.name):
        core.resolve_terrain_mode(True, True)
    assert "Both LiDAR and DEM terrain were requested" in caplog.text


@pytest.mark.parametrize("lidar, dem", [(False, False), (True, False), (False, True)])
def test_single_terrain_choice_does_not_warn(lidar, dem, caplog):
    with caplog.at_level(logging.WARNING, logger=core.logger.name):
        core.resolve_terrain_mode(lidar, dem)
    assert caplog.records == []


@pytest.mark.parametrize(
    "generate_roads, source, terrain_mode, expected, warning",
    [
        (False, "overture", None, False, None),
        (True, "overture", None, True, None),
        (True, "osm", None, False, "only supported with --building-data-source overture"),
        (True, "overture", core.TERRAIN_MODE_DEM, False, "not supported with DEM terrain"),
        (True, "overture", core.TERRAIN_MODE_LIDAR, False, "not supported with LiDAR terrain"),
        (False, "osm", core.TERRAIN_MODE_DEM, False, None),  # roads not requested: stay quiet
    ],
)
def test_resolve_roads_enabled(generate_roads, source, terrain_mode, expected, warning, caplog):
    with caplog.at_level(logging.WARNING, logger=core.logger.name):
        assert core.resolve_roads_enabled(generate_roads, source, terrain_mode) is expected
    if warning is None:
        assert caplog.records == []
    else:
        assert warning in caplog.text
