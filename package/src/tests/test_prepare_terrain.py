"""
core.prepare_terrain_and_hag: only fetch what the enabled features need, and
turn terrain failures into a clear RuntimeError.

The key regression is test_dem_with_calibration_survives_hag_failure: with
the old code, DEM + LiDAR crashed with "TIF file does not exist" even though
the DEM mesh never needs the LiDAR HAG raster.
"""
import logging
from pathlib import Path

import pytest

from scene_generation import core


def test_dem_only_builds_dem_and_never_downloads_lidar(fake_lidar, fake_dem_mesh, terrain_kwargs):
    ply, hag = core.prepare_terrain_and_hag(
        terrain_mode=core.TERRAIN_MODE_DEM, lidar_height_calibration=False, **terrain_kwargs
    )
    assert Path(ply).name == core.DEM_TERRAIN_PLY and Path(ply).exists()
    assert hag is None
    assert fake_lidar.calls == []
    assert len(fake_dem_mesh.calls) == 1


def test_dem_mesh_gets_scene_crs_and_centre(fake_lidar, fake_dem_mesh, terrain_kwargs):
    """Downstream 1: the DEM mesh must be localized to the same centre as the buildings."""
    core.prepare_terrain_and_hag(
        terrain_mode=core.TERRAIN_MODE_DEM, lidar_height_calibration=False, **terrain_kwargs
    )
    kwargs = fake_dem_mesh.calls[0]["kwargs"]
    assert kwargs["utm_crs"] == terrain_kwargs["utm_crs"]
    assert kwargs["center_x"] == terrain_kwargs["center_x"]
    assert kwargs["center_y"] == terrain_kwargs["center_y"]


def test_dem_with_calibration_survives_hag_failure(fake_lidar, fake_dem_mesh, terrain_kwargs, caplog):
    """The originally reported crash: HAG raster step fails, DEM terrain must still be built."""
    fake_lidar.mode = "laz_only"
    with caplog.at_level(logging.WARNING, logger=core.logger.name):
        ply, hag = core.prepare_terrain_and_hag(
            terrain_mode=core.TERRAIN_MODE_DEM, lidar_height_calibration=True, **terrain_kwargs
        )
    assert Path(ply).name == core.DEM_TERRAIN_PLY and Path(ply).exists()
    assert hag is None
    assert "height calibration is disabled" in caplog.text
    assert "Unable to write empty raster" in caplog.text  # the real PDAL error is surfaced


def test_dem_with_calibration_and_no_lidar_coverage_still_builds_dem(
    fake_lidar, fake_dem_mesh, terrain_kwargs
):
    fake_lidar.mode = "no_cover"
    ply, hag = core.prepare_terrain_and_hag(
        terrain_mode=core.TERRAIN_MODE_DEM, lidar_height_calibration=True, **terrain_kwargs
    )
    assert Path(ply).exists() and hag is None


def test_lidar_download_is_asked_to_raise(fake_lidar, fake_lidar_mesh, terrain_kwargs):
    core.prepare_terrain_and_hag(
        terrain_mode=core.TERRAIN_MODE_LIDAR, lidar_height_calibration=False, **terrain_kwargs
    )
    assert fake_lidar.calls[0]["raise_on_error"] is True


def test_lidar_terrain_builds_from_laz_even_if_hag_raster_fails(
    fake_lidar, fake_lidar_mesh, terrain_kwargs
):
    """LiDAR terrain only needs the .laz; a failed .tif must not stop it."""
    fake_lidar.mode = "laz_only"
    ply, hag = core.prepare_terrain_and_hag(
        terrain_mode=core.TERRAIN_MODE_LIDAR, lidar_height_calibration=True, **terrain_kwargs
    )
    assert Path(ply).name == core.LIDAR_TERRAIN_PLY and Path(ply).exists()
    assert hag is None
    assert Path(fake_lidar_mesh.calls[0]["src"]).name == "test_hag.laz"


def test_lidar_terrain_without_coverage_raises_with_original_error(
    fake_lidar, fake_lidar_mesh, terrain_kwargs
):
    fake_lidar.mode = "no_cover"
    with pytest.raises(RuntimeError, match="No LiDAR data available"):
        core.prepare_terrain_and_hag(
            terrain_mode=core.TERRAIN_MODE_LIDAR, lidar_height_calibration=False, **terrain_kwargs
        )
    assert fake_lidar_mesh.calls == []


def test_dem_mesh_failure_raises_runtime_error(fake_lidar, fake_dem_mesh, terrain_kwargs):
    fake_dem_mesh.fail_with = RuntimeError("No USGS 1 m DEM tiles cover this area.")
    with pytest.raises(RuntimeError, match="DEM terrain generation failed.*No USGS 1 m DEM tiles"):
        core.prepare_terrain_and_hag(
            terrain_mode=core.TERRAIN_MODE_DEM, lidar_height_calibration=False, **terrain_kwargs
        )


def test_calibration_returns_generated_tif_path(fake_lidar, terrain_kwargs):
    """Bug #2: the HAG path must come back to the caller (the OSM branch relies on this)."""
    ply, hag = core.prepare_terrain_and_hag(
        terrain_mode=None, lidar_height_calibration=True, **terrain_kwargs
    )
    assert ply is None
    assert Path(hag) == Path(terrain_kwargs["data_dir"]) / "test_hag.tif"


def test_existing_tif_is_reused_without_download(fake_lidar, terrain_kwargs):
    (Path(terrain_kwargs["data_dir"]) / "test_hag.tif").write_bytes(b"tif")
    _, hag = core.prepare_terrain_and_hag(
        terrain_mode=None, lidar_height_calibration=True, **terrain_kwargs
    )
    assert Path(hag).name == "test_hag.tif"
    assert fake_lidar.calls == []


def test_user_supplied_hag_path_is_kept(fake_lidar, terrain_kwargs):
    terrain_kwargs["hag_tiff_path"] = "/somewhere/my_hag.tif"
    _, hag = core.prepare_terrain_and_hag(
        terrain_mode=None, lidar_height_calibration=True, **terrain_kwargs
    )
    assert hag == "/somewhere/my_hag.tif"
    assert fake_lidar.calls == []


def test_flat_ground_without_calibration_fetches_nothing(fake_lidar, fake_dem_mesh, terrain_kwargs):
    assert core.prepare_terrain_and_hag(
        terrain_mode=None, lidar_height_calibration=False, **terrain_kwargs
    ) == (None, None)
    assert fake_lidar.calls == [] and fake_dem_mesh.calls == []
