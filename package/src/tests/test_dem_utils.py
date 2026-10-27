"""
Downstream 1: DEM mesh alignment and the UTM zone-edge fix.

* dem_to_ply must put the scene centre at x = y = 0 with the ground there at
  z = 0, whatever cells are missing (old code centred on the mesh itself).
* clip_reproject_dem_to_wgs84_utm in single_zone mode must put every tile into
  the scene's UTM zone and write one dem.tif (old code split zones and left
  the second zone in a random dem_<uuid>.tif).
"""
import re

import numpy as np
import pytest
import rasterio
import trimesh
from pyproj import Transformer

from scene_generation import dem_utils

from conftest import NODATA, write_dem


# ---------------------------------------------------------------------------
# dem_to_ply
# ---------------------------------------------------------------------------
def _mesh(sloped_dem, tmp_path, **kwargs):
    path, info = sloped_dem
    out = tmp_path / "out.ply"
    dem_utils.dem_to_ply(str(path), str(out), **kwargs)
    return trimesh.load(out, process=False).vertices, info


def test_scene_centre_lands_at_origin(sloped_dem, tmp_path):
    _, info = sloped_dem
    v, _ = _mesh(sloped_dem, tmp_path, center_x=info.cx, center_y=info.cy)
    at_centre = v[np.hypot(v[:, 0], v[:, 1]) < 1e-6]
    assert len(at_centre) == 1, "a vertex should sit exactly on the scene centre"
    assert at_centre[0, 2] == pytest.approx(0.0, abs=1e-6)


def test_ground_heights_are_relative_to_the_centre(sloped_dem, tmp_path):
    """Slope is 0.1 m/m east, so 50 m east of the centre must be 5 m up."""
    _, info = sloped_dem
    v, _ = _mesh(sloped_dem, tmp_path, center_x=info.cx, center_y=info.cy)
    pt = v[(np.abs(v[:, 0] - 50) < 1e-6) & (np.abs(v[:, 1]) < 1e-6)]
    assert pt[0, 2] == pytest.approx(5.0, abs=1e-6)


def test_missing_cells_do_not_shift_the_mesh(sloped_dem, tmp_path):
    """The 20 m nodata strip on the east edge must not move the centre (old bug)."""
    _, info = sloped_dem
    v, _ = _mesh(sloped_dem, tmp_path, center_x=info.cx, center_y=info.cy)
    # cells are 0..179 valid, centre is cell 100 -> x spans -100 .. +79
    assert v[:, 0].min() == pytest.approx(-100.0)
    assert v[:, 0].max() == pytest.approx(79.0)


def test_vertices_sit_at_pixel_centres(sloped_dem, tmp_path):
    _, info = sloped_dem
    v, _ = _mesh(sloped_dem, tmp_path, center_x=info.cx, center_y=info.cy)
    abs_x = v[:, 0] + info.cx
    # pixel centres are at x0 + 0.5, x0 + 1.5, ...
    assert np.allclose((abs_x - info.x0) % 1.0, 0.5)


def test_nodata_never_becomes_a_vertex(sloped_dem, tmp_path):
    _, info = sloped_dem
    v, _ = _mesh(sloped_dem, tmp_path, center_x=info.cx, center_y=info.cy)
    assert np.abs(v[:, 2]).max() < 100


def test_legacy_call_without_centre_still_works(sloped_dem, tmp_path):
    v, _ = _mesh(sloped_dem, tmp_path)
    assert np.allclose(v.mean(axis=0), 0.0, atol=1e-6)


def test_dem_without_valid_cells_raises(tmp_path):
    path = write_dem(tmp_path / "empty.tif", np.full((10, 10), NODATA), "EPSG:32619", 0.0, 10.0)
    with pytest.raises(ValueError, match="no valid cells"):
        dem_utils.dem_to_ply(str(path), str(tmp_path / "out.ply"), center_x=5, center_y=5)


# ---------------------------------------------------------------------------
# clip_reproject_dem_to_wgs84_utm
# ---------------------------------------------------------------------------
# A 400 m box straddling the 72°W boundary between UTM 18N and 19N.
AOI_LONLAT = (-72.0025, 42.3500, -71.9975, 42.3530)


def _tile_around_aoi(path, crs, value):
    """Write a 600 m DEM tile in `crs` that fully covers the AOI."""
    import shapely

    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    cx, cy = t.transform((AOI_LONLAT[0] + AOI_LONLAT[2]) / 2, (AOI_LONLAT[1] + AOI_LONLAT[3]) / 2)
    return write_dem(path, np.full((150, 150), value), crs, cx - 300, cy + 300, res=4.0)


@pytest.fixture
def aoi():
    import shapely

    return shapely.box(*AOI_LONLAT)


def test_tile_from_another_zone_is_reprojected_into_target_zone(tmp_path, aoi):
    tile = _tile_around_aoi(tmp_path / "zone18.tif", "EPSG:32618", 12.0)
    res = dem_utils.clip_reproject_dem_to_wgs84_utm(
        aoi, [str(tile)], mode="single_zone", target_epsg=32619, mosaic=True,
        out_prefix=str(tmp_path / "dem"), overwrite=True,
    )
    assert res["path"] == str(tmp_path / "dem.tif")
    with rasterio.open(res["path"]) as ds:
        assert ds.crs.to_epsg() == 32619
        data = ds.read(1)
        assert np.isclose(data, 12.0).any(), "reprojected DEM has no valid cells"


def test_tiles_from_two_zones_end_up_in_one_dem(tmp_path, aoi):
    t18 = _tile_around_aoi(tmp_path / "zone18.tif", "EPSG:32618", 12.0)
    t19 = _tile_around_aoi(tmp_path / "zone19.tif", "EPSG:32619", 12.0)
    res = dem_utils.clip_reproject_dem_to_wgs84_utm(
        aoi, [str(t18), str(t19)], mode="single_zone", target_epsg=32619, mosaic=True,
        out_prefix=str(tmp_path / "dem"), overwrite=True,
    )
    assert res["path"] == str(tmp_path / "dem.tif")
    assert res["skipped_sources"] == []
    with rasterio.open(res["path"]) as ds:
        assert ds.crs.to_epsg() == 32619


def test_rerun_overwrites_dem_tif_instead_of_writing_a_uuid_copy(tmp_path, aoi):
    tile = _tile_around_aoi(tmp_path / "zone19.tif", "EPSG:32619", 12.0)
    kwargs = dict(mode="single_zone", target_epsg=32619, mosaic=True,
                  out_prefix=str(tmp_path / "dem"), overwrite=True)
    dem_utils.clip_reproject_dem_to_wgs84_utm(aoi, [str(tile)], **kwargs)
    res = dem_utils.clip_reproject_dem_to_wgs84_utm(aoi, [str(tile)], **kwargs)
    assert res["path"] == str(tmp_path / "dem.tif")
    stray = [p.name for p in tmp_path.iterdir() if re.fullmatch(r"dem_[0-9a-f]{8}\.tif", p.name)]
    assert stray == []


def test_debug_logging_does_not_recurse(tmp_path, aoi, capsys):
    tile = _tile_around_aoi(tmp_path / "zone19.tif", "EPSG:32619", 12.0)
    dem_utils.clip_reproject_dem_to_wgs84_utm(
        aoi, [str(tile)], mode="single_zone", target_epsg=32619,
        out_prefix=str(tmp_path / "dem"), overwrite=True, debug=True,
    )
    assert "[DBG]" in capsys.readouterr().out
