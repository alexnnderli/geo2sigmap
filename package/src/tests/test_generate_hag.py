"""
USGS_LiDAR_HAG.generate_hag: with raise_on_error=True it must surface the real
PDAL error, and must not report success when its outputs were not written.

Before the fix it printed the PDAL error and returned normally, so the caller
only saw a misleading "TIF file does not exist" later on.

PDAL itself is replaced by a fake, and the 3DEP boundary file is a small local
GeoJSON, so nothing is downloaded.
"""
import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import shapely


class FakePipeline:
    behaviour = "write_both"   # "write_both" | "write_nothing" | "raise"
    last_json = None

    def __init__(self, pipeline_json):
        FakePipeline.last_json = json.loads(pipeline_json)

    def execute(self):
        if self.behaviour == "raise":
            raise RuntimeError("writers.gdal: Unable to write empty raster")
        if self.behaviour == "write_both":
            for stage in self.last_json["pipeline"]:
                if stage.get("type", "").startswith("writers."):
                    Path(stage["filename"]).write_bytes(b"x")


@pytest.fixture
def hag_module(monkeypatch, tmp_path):
    """Import USGS_LiDAR_HAG fresh with a fake `pdal`, inside an empty working dir."""
    monkeypatch.setitem(sys.modules, "pdal", types.SimpleNamespace(Pipeline=FakePipeline))
    monkeypatch.delitem(sys.modules, "scene_generation.USGS_LiDAR_HAG", raising=False)
    mod = importlib.import_module("scene_generation.USGS_LiDAR_HAG")
    monkeypatch.delitem(sys.modules, "scene_generation.USGS_LiDAR_HAG", raising=False)

    # generate_hag reads resources.geojson from the current directory
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    boundary = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {"name": "FAKE_3DEP_PROJECT", "url": "https://example.invalid/ept.json", "count": 1},
            "geometry": shapely.geometry.mapping(shapely.box(-72.0, 42.0, -70.0, 43.0)),
        }],
    }
    (work / "resources.geojson").write_text(json.dumps(boundary))
    FakePipeline.behaviour = "write_both"
    return mod


@pytest.fixture
def out_dir(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    return d


def test_success_writes_laz_and_tif(hag_module, out_dir, scene_geometry):
    hag_module.generate_hag(scene_geometry.polygon_4326, str(out_dir), scene_geometry.utm_crs,
                            raise_on_error=True)
    assert (out_dir / "test_hag.laz").exists()
    assert (out_dir / "test_hag.tif").exists()


def test_pdal_error_is_reraised(hag_module, out_dir, scene_geometry):
    FakePipeline.behaviour = "raise"
    with pytest.raises(RuntimeError, match="Unable to write empty raster"):
        hag_module.generate_hag(scene_geometry.polygon_4326, str(out_dir), scene_geometry.utm_crs,
                                raise_on_error=True)


def test_missing_outputs_raise_and_name_the_file(hag_module, out_dir, scene_geometry):
    FakePipeline.behaviour = "write_nothing"
    with pytest.raises(RuntimeError, match="test_hag.tif"):
        hag_module.generate_hag(scene_geometry.polygon_4326, str(out_dir), scene_geometry.utm_crs,
                                raise_on_error=True)


def test_legacy_mode_does_not_raise(hag_module, out_dir, scene_geometry):
    """raise_on_error=False keeps the old print-and-return behaviour for other callers."""
    FakePipeline.behaviour = "raise"
    assert hag_module.generate_hag(scene_geometry.polygon_4326, str(out_dir),
                                   scene_geometry.utm_crs) is None


def test_pipeline_reprojects_to_the_scene_crs(hag_module, out_dir, scene_geometry):
    hag_module.generate_hag(scene_geometry.polygon_4326, str(out_dir), scene_geometry.utm_crs,
                            raise_on_error=True)
    reproj = [s for s in FakePipeline.last_json["pipeline"] if s.get("type") == "filters.reprojection"]
    assert reproj and reproj[0]["out_srs"] == scene_geometry.utm_crs.to_string()


def test_no_coverage_raises_value_error(hag_module, out_dir):
    far_away = shapely.box(-120.0, 35.0, -119.99, 35.01)
    with pytest.raises(ValueError, match="No LiDAR data available"):
        hag_module.generate_hag(far_away, str(out_dir), "EPSG:32611", raise_on_error=True)
