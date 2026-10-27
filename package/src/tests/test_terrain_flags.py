"""
Flag-combination tests for terrain / HAG / roads handling.

Covers the edge cases from the bug report:
  1. Roads (Overture) combined with DEM or LiDAR terrain
  2. DEM and LiDAR terrain enabled together (previously crashed on a
     missing HAG/LAZ file)
plus DEM-only, LiDAR-only and the HAG retry/raise_on_error behaviour.
"""
import json
import logging

import pyproj
import pytest
from shapely.geometry import Polygon, box, mapping

import scene_generation.USGS_LiDAR_HAG as hag_mod
import scene_generation.core as core
import scene_generation.lidar_terrain_mesh as lidar_mod

UTM = pyproj.CRS.from_epsg(32619)  # Boston
AOI = Polygon([(-71.06, 42.351), (-71.06, 42.359), (-71.048, 42.359), (-71.048, 42.351)])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
class Calls:
    def __init__(self):
        self.hag = 0
        self.dem = []
        self.lidar = 0


@pytest.fixture
def env(tmp_path, monkeypatch):
    data_dir = tmp_path / "scene"
    mesh_dir = data_dir / "mesh"
    mesh_dir.mkdir(parents=True)
    calls = Calls()
    state = {"hag_fails": False, "hag_writes": ("laz", "tif")}

    def fake_generate_hag(polygon, data_dir_arg, crs, raise_on_error=True, **kw):
        calls.hag += 1
        if state["hag_fails"]:
            raise RuntimeError("HAG generation failed after 3 attempt(s): S3 timeout")
        for ext in state["hag_writes"]:
            (data_dir / f"test_hag.{ext}").write_text("x")
        return True

    def fake_dem(aoi, ply_path, utm_crs=None):
        calls.dem.append(utm_crs)
        if state.get("dem_fails"):
            raise RuntimeError("No USGS 1 m DEM coverage found")
        open(ply_path, "w").write("ply")

    def fake_lidar(laz, ply, **kw):
        calls.lidar += 1
        open(ply, "w").write("ply")

    monkeypatch.setattr(hag_mod, "generate_hag", fake_generate_hag)
    monkeypatch.setattr(core, "generate_terrain_mesh_dem", fake_dem)
    monkeypatch.setattr(lidar_mod, "generate_terrain_mesh", fake_lidar)

    def run(lidar_terrain=False, dem_terrain=False, calibration=False, hag_tiff_path=None):
        mode = core.resolve_terrain_mode(lidar_terrain, dem_terrain)
        return core.prepare_terrain_and_hag(
            terrain_mode=mode,
            lidar_height_calibration=calibration,
            hag_tiff_path=hag_tiff_path,
            data_dir=data_dir,
            mesh_data_dir=mesh_dir,
            ground_polygon_4326=AOI,
            ground_scale=1.5,
            utm_crs=UTM,
            center_x=0.0,
            center_y=0.0,
        )

    return run, calls, state, data_dir, mesh_dir


# --------------------------------------------------------------------------
# flag resolution
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "lidar,dem,expected",
    [(False, False, None), (True, False, "lidar"), (False, True, "dem"), (True, True, "dem")],
)
def test_resolve_terrain_mode(lidar, dem, expected):
    assert core.resolve_terrain_mode(lidar, dem) == expected


def test_both_terrain_flags_warn(caplog):
    with caplog.at_level(logging.WARNING):
        core.resolve_terrain_mode(True, True)
    assert "Both LiDAR and DEM terrain" in caplog.text


@pytest.mark.parametrize("mode", ["lidar", "dem"])
def test_edge_case_1_roads_with_terrain_warns_and_skips(mode, caplog):
    with caplog.at_level(logging.WARNING):
        enabled = core.resolve_roads_enabled(True, "overture", mode)
    assert enabled is False
    assert "roads will NOT be generated" in caplog.text


def test_roads_with_osm_warns_and_skips(caplog):
    with caplog.at_level(logging.WARNING):
        assert core.resolve_roads_enabled(True, "osm", None) is False
    assert "only supported with --building-data-source overture" in caplog.text


def test_roads_flat_overture_enabled_without_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert core.resolve_roads_enabled(True, "overture", None) is True
        assert core.resolve_roads_enabled(False, "overture", "dem") is False
    assert caplog.text == ""


# --------------------------------------------------------------------------
# terrain / HAG preparation
# --------------------------------------------------------------------------
def test_dem_only_works_without_lidar_and_skips_hag(env):
    run, calls, state, data_dir, mesh_dir = env
    ply, hag = run(dem_terrain=True)
    assert ply == mesh_dir / "dem_terrain.ply" and ply.exists()
    assert calls.hag == 0, "DEM mode must not download the LiDAR point cloud"
    assert calls.dem == [UTM], "DEM must be reprojected into the scene's UTM zone"
    assert hag is None


def test_edge_case_2_both_terrain_flags_hag_failure_does_not_crash(env, caplog):
    """The reported crash: DEM + LiDAR + calibration with a failing HAG step."""
    run, calls, state, data_dir, mesh_dir = env
    state["hag_fails"] = True
    with caplog.at_level(logging.WARNING):
        ply, hag = run(lidar_terrain=True, dem_terrain=True, calibration=True)
    assert ply == mesh_dir / "dem_terrain.ply" and ply.exists()
    assert hag is None
    assert calls.lidar == 0
    assert "height calibration is disabled" in caplog.text


def test_both_terrain_flags_without_calibration_never_touches_hag(env):
    run, calls, *_ = env
    run(lidar_terrain=True, dem_terrain=True)
    assert calls.hag == 0


def test_dem_with_calibration_uses_generated_tif(env):
    run, calls, state, data_dir, mesh_dir = env
    ply, hag = run(dem_terrain=True, calibration=True)
    assert calls.hag == 1
    assert hag == str(data_dir / "test_hag.tif")


def test_lidar_terrain_hag_failure_raises_clear_error(env):
    run, calls, state, *_ = env
    state["hag_fails"] = True
    with pytest.raises(RuntimeError, match="LiDAR terrain was requested but the USGS LiDAR download failed"):
        run(lidar_terrain=True)


def test_lidar_terrain_only_needs_laz_not_tif(env):
    """Existing LAZ + failing re-download for the TIF must not block the LiDAR mesh."""
    run, calls, state, data_dir, mesh_dir = env
    (data_dir / "test_hag.laz").write_text("x")
    state["hag_fails"] = True
    ply, hag = run(lidar_terrain=True, calibration=True)
    assert ply == mesh_dir / "lidar_terrain.ply" and calls.lidar == 1
    assert hag is None


def test_lidar_terrain_success(env):
    run, calls, state, data_dir, mesh_dir = env
    ply, _ = run(lidar_terrain=True)
    assert calls.hag == 1 and calls.lidar == 1 and ply.name == "lidar_terrain.ply"


def test_dem_failure_raises(env):
    run, calls, state, *_ = env
    state["dem_fails"] = True
    with pytest.raises(RuntimeError, match="DEM terrain generation failed"):
        run(dem_terrain=True)


def test_cached_lidar_mesh_is_not_reused_for_dem(env):
    """Switching from LiDAR to DEM must build a DEM mesh, not reuse lidar_terrain.ply."""
    run, calls, state, data_dir, mesh_dir = env
    (mesh_dir / "lidar_terrain.ply").write_text("old lidar mesh")
    ply, _ = run(dem_terrain=True)
    assert ply.name == "dem_terrain.ply" and len(calls.dem) == 1


def test_no_terrain_no_calibration_does_nothing(env):
    run, calls, *_ = env
    assert run() == (None, None)
    assert calls.hag == 0 and calls.dem == [] and calls.lidar == 0


# --------------------------------------------------------------------------
# generate_hag: raise_on_error + retries (real implementation, fake PDAL)
# --------------------------------------------------------------------------
@pytest.fixture
def hag_env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "MA_TEST", "url": "u", "count": 1},
            "geometry": mapping(box(-72, 42, -70, 43)),
        }],
    }
    (tmp_path / "resources.geojson").write_text(json.dumps(geojson))
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr(hag_mod.time, "sleep", lambda s: None)

    attempts = {"n": 0, "fail_first": 0, "partial": False}

    class FakePipeline:
        def __init__(self, spec):
            pass

        def execute(self):
            attempts["n"] += 1
            if attempts["n"] <= attempts["fail_first"]:
                if attempts["partial"]:
                    (out / "test_hag.laz").write_text("partial")
                raise RuntimeError("Unable to fetch EPT data")
            (out / "test_hag.laz").write_text("laz")
            (out / "test_hag.tif").write_text("tif")

    monkeypatch.setattr(hag_mod.pdal, "Pipeline", FakePipeline)
    return attempts, out


def _real_generate_hag():
    # The `env` fixture is not used here, so this is the real function.
    return hag_mod.generate_hag


def test_generate_hag_retries_then_succeeds(hag_env):
    attempts, out = hag_env
    attempts["fail_first"] = 2
    assert _real_generate_hag()(AOI, str(out), UTM, max_retries=3, retry_delay_s=0) is True
    assert attempts["n"] == 3
    assert (out / "test_hag.laz").exists() and (out / "test_hag.tif").exists()


def test_generate_hag_raises_after_all_retries_and_cleans_partial_files(hag_env):
    attempts, out = hag_env
    attempts["fail_first"] = 99
    attempts["partial"] = True
    with pytest.raises(RuntimeError, match=r"HAG generation failed after 3 attempt\(s\).*EPT"):
        _real_generate_hag()(AOI, str(out), UTM, max_retries=3, retry_delay_s=0)
    assert attempts["n"] == 3
    assert not (out / "test_hag.laz").exists(), "partial LAZ must not look like a success"


def test_generate_hag_raise_on_error_false_returns_false(hag_env, capsys):
    attempts, out = hag_env
    attempts["fail_first"] = 99
    assert _real_generate_hag()(AOI, str(out), UTM, raise_on_error=False, max_retries=2, retry_delay_s=0) is False
    assert "Error generating HAG data" in capsys.readouterr().out


def test_generate_hag_no_coverage_is_not_retried(hag_env):
    attempts, out = hag_env
    far_away = box(-100, 30, -99.99, 30.01)
    with pytest.raises(ValueError, match="No USGS 3DEP LiDAR dataset"):
        _real_generate_hag()(far_away, str(out), UTM, retry_delay_s=0)
    assert attempts["n"] == 0


def test_generate_hag_accepts_string_crs(hag_env):
    attempts, out = hag_env
    assert _real_generate_hag()(AOI, str(out), "EPSG:32619", retry_delay_s=0) is True
