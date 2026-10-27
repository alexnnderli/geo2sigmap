"""
Scene() end to end, offline: checks that Scene actually USES the new terrain
logic (not just that the helper functions exist).

Downloads and mesh builders are faked (see conftest.py). Building/road data
sources are replaced with empty results, so a run writes a real scene.xml with
ground only. Tests that only care about the terrain step stop the run right
after it by making the first triangulation raise.
"""
import logging
import types
from pathlib import Path

import geopandas as gpd
import pytest

from scene_generation import core

from conftest import BOSTON_POINTS


class StopAfterTerrain(Exception):
    pass


@pytest.fixture
def offline_scene(monkeypatch, fake_lidar, fake_dem_mesh, fake_lidar_mesh):
    """Fake every network source; return a recorder of what Scene read."""
    rec = types.SimpleNamespace(pv_reads=[], hag_paths=[])

    def fake_pv_read(path):
        rec.pv_reads.append(Path(path).name)
        return types.SimpleNamespace(bounds=(0, 0, 0, 0, 0, 0))

    class FakeHag:
        def __init__(self, path):
            rec.hag_paths.append(path)

        def query(self, *a, **k):
            raise AssertionError("no buildings, so HAG should never be queried")

    empty = lambda *a, **k: gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    monkeypatch.setattr(core.pv, "read", fake_pv_read)
    monkeypatch.setattr(core, "GeoTIFFHandler", FakeHag)
    monkeypatch.setattr(core, "load_overture_buildings_for_aoi", empty)
    monkeypatch.setattr(core, "load_overture_building_parts_for_aoi", empty)
    monkeypatch.setattr(core.ox.features, "features_from_bbox", empty)
    rec.lidar, rec.dem_mesh, rec.lidar_mesh = fake_lidar, fake_dem_mesh, fake_lidar_mesh
    return rec


@pytest.fixture
def stop_after_terrain(monkeypatch):
    def boom(*a, **k):
        raise StopAfterTerrain()

    monkeypatch.setattr(core, "triangulate", boom)


def run(tmp_path, **flags):
    kwargs = dict(lidar_height_calibration=False, generate_building_map=False)
    kwargs.update(flags)
    return core.Scene()(BOSTON_POINTS, str(tmp_path), None, **kwargs)


# ---------------------------------------------------------------------------
# Main bug 1: DEM terrain no longer depends on LiDAR
# ---------------------------------------------------------------------------
def test_dem_only_builds_and_uses_dem_terrain(offline_scene, stop_after_terrain, tmp_path):
    with pytest.raises(StopAfterTerrain):
        run(tmp_path, dem_terrain=True)
    assert [c["ply"].name for c in offline_scene.dem_mesh.calls] == [core.DEM_TERRAIN_PLY]
    assert offline_scene.pv_reads == [core.DEM_TERRAIN_PLY]
    assert offline_scene.lidar.calls == []


def test_dem_plus_lidar_does_not_download_lidar(offline_scene, stop_after_terrain, tmp_path):
    with pytest.raises(StopAfterTerrain):
        run(tmp_path, dem_terrain=True, lidar_terrain=True)
    assert offline_scene.lidar.calls == []
    assert offline_scene.lidar_mesh.calls == []
    assert offline_scene.pv_reads == [core.DEM_TERRAIN_PLY]


def test_dem_plus_lidar_survives_hag_raster_failure(offline_scene, stop_after_terrain, tmp_path):
    """The originally reported crash ("TIF file does not exist")."""
    offline_scene.lidar.mode = "laz_only"
    with pytest.raises(StopAfterTerrain):
        run(tmp_path, dem_terrain=True, lidar_terrain=True, lidar_height_calibration=True)
    assert offline_scene.pv_reads == [core.DEM_TERRAIN_PLY]


def test_terrain_failure_raises_instead_of_returning_1(offline_scene, tmp_path):
    offline_scene.dem_mesh.fail_with = RuntimeError("No USGS 1 m DEM tiles cover this area.")
    with pytest.raises(RuntimeError, match="DEM terrain generation failed"):
        run(tmp_path, dem_terrain=True)


def test_scene_xml_points_at_the_dem_mesh(offline_scene, tmp_path):
    run(tmp_path, dem_terrain=True)
    xml = (tmp_path / "scene.xml").read_text()
    assert f"mesh/{core.DEM_TERRAIN_PLY}" in xml
    assert "mesh/lidar_terrain.ply" not in xml


def test_flat_scene_xml_uses_ground_ply(offline_scene, tmp_path):
    run(tmp_path)
    assert "mesh/ground.ply" in (tmp_path / "scene.xml").read_text()


# ---------------------------------------------------------------------------
# Main bug 2: dropped roads are reported
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "flags, warning",
    [
        (dict(dem_terrain=True), "not supported with DEM terrain"),
        (dict(lidar_terrain=True), "not supported with LiDAR terrain"),
        (dict(building_data_source="osm"), "only supported with --building-data-source overture"),
    ],
)
def test_skipped_roads_are_warned_about(offline_scene, stop_after_terrain, tmp_path, caplog, flags, warning):
    with caplog.at_level(logging.WARNING, logger=core.logger.name), pytest.raises(StopAfterTerrain):
        run(tmp_path, generate_roads=True, **flags)
    assert warning in caplog.text


# ---------------------------------------------------------------------------
# Known-issue #2: OSM height calibration gets the generated HAG raster
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("source", ["osm", "overture"])
def test_height_calibration_uses_generated_hag(offline_scene, tmp_path, source):
    run(tmp_path, building_data_source=source, lidar_height_calibration=True)
    assert offline_scene.hag_paths == [str(tmp_path / "test_hag.tif")]
