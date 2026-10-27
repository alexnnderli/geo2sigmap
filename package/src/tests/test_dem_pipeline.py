"""
dem.generate_terrain_mesh_dem: wires the scene CRS and centre through, always
mosaics into one zone, and fails loudly instead of returning nothing.
The USGS locator, download and mesher are faked.
"""
import types
from pathlib import Path

import pytest

from scene_generation import dem


@pytest.fixture
def fake_dem_deps(monkeypatch, tmp_path):
    rec = types.SimpleNamespace(links=["https://example.invalid/tile.tif"], clip_kwargs=None,
                                ply_kwargs=None, write_tif=True, write_ply=True)

    class FakeLocator:
        def __init__(self, fesm, grid, debug=True):
            pass

        def set_debug(self, flag):
            pass

        def get_links(self, aoi):
            return rec.links

    def fake_clip(aoi, links, **kwargs):
        rec.clip_kwargs = kwargs
        path = Path(kwargs["out_prefix"] + ".tif")
        if rec.write_tif:
            path.write_bytes(b"tif")
            return {"path": str(path), "skipped_sources": []}
        return {"path": None, "skipped_sources": links}

    def fake_dem_to_ply(tif, ply, **kwargs):
        rec.ply_kwargs = kwargs
        if rec.write_ply:
            Path(ply).write_bytes(b"ply")

    monkeypatch.setattr(dem, "get_fesm_paths", lambda: [tmp_path / "grid.gpkg", tmp_path / "fesm.gpkg"])
    monkeypatch.setattr(dem, "USGS1mLocator", FakeLocator)
    monkeypatch.setattr(dem, "clip_reproject_dem_to_wgs84_utm", fake_clip)
    monkeypatch.setattr(dem, "dem_to_ply", fake_dem_to_ply)
    return rec


def _run(tmp_path, scene_geometry):
    return dem.generate_terrain_mesh_dem(
        scene_geometry.polygon_4326, tmp_path / "mesh" / "dem_terrain.ply",
        utm_crs=scene_geometry.utm_crs,
        center_x=scene_geometry.center_x, center_y=scene_geometry.center_y,
    )


def test_uses_single_zone_in_the_scene_crs(fake_dem_deps, tmp_path, scene_geometry):
    _run(tmp_path, scene_geometry)
    kw = fake_dem_deps.clip_kwargs
    assert kw["mode"] == "single_zone"
    assert kw["target_epsg"] == scene_geometry.utm_crs.to_epsg()
    assert kw["mosaic"] is True and kw["overwrite"] is True


def test_passes_scene_centre_to_the_mesher(fake_dem_deps, tmp_path, scene_geometry):
    out = _run(tmp_path, scene_geometry)
    assert Path(out).exists()
    assert fake_dem_deps.ply_kwargs["center_x"] == scene_geometry.center_x
    assert fake_dem_deps.ply_kwargs["center_y"] == scene_geometry.center_y


def test_no_tiles_raises(fake_dem_deps, tmp_path, scene_geometry):
    fake_dem_deps.links = []
    with pytest.raises(RuntimeError, match="No USGS 1 m DEM tiles"):
        _run(tmp_path, scene_geometry)


def test_no_raster_raises(fake_dem_deps, tmp_path, scene_geometry):
    fake_dem_deps.write_tif = False
    with pytest.raises(RuntimeError, match="produced no raster"):
        _run(tmp_path, scene_geometry)


def test_no_mesh_raises(fake_dem_deps, tmp_path, scene_geometry):
    fake_dem_deps.write_ply = False
    with pytest.raises(RuntimeError, match="DEM mesh was not written"):
        _run(tmp_path, scene_geometry)
