"""
Shared fixtures for the scene_generation test suite.

Everything here runs offline: LiDAR/DEM downloads and mesh builders are
replaced with small fakes that record how they were called. Tests marked
``network`` hit the real USGS/Overture services and are skipped unless you
pass ``--run-network``.
"""
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


# ---------------------------------------------------------------------------
# --run-network option
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="also run tests that download real USGS / Overture data (slow)",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "network: downloads real data; needs --run-network")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-network"):
        return
    skip = pytest.mark.skip(reason="needs --run-network")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Scene geometry shared by several tests (Boston test bbox from the README)
# ---------------------------------------------------------------------------
BOSTON_BBOX = (-71.0602, 42.3512, -71.0484, 42.3591)
BOSTON_POINTS = [
    [BOSTON_BBOX[0], BOSTON_BBOX[1]],
    [BOSTON_BBOX[0], BOSTON_BBOX[3]],
    [BOSTON_BBOX[2], BOSTON_BBOX[3]],
    [BOSTON_BBOX[2], BOSTON_BBOX[1]],
    [BOSTON_BBOX[0], BOSTON_BBOX[1]],
]


@pytest.fixture
def scene_geometry():
    """Ground polygon, UTM CRS and scene centre, computed the same way as core.Scene."""
    import shapely
    from pyproj import Transformer

    from scene_generation.utils import get_utm_epsg_code_from_gps

    utm = get_utm_epsg_code_from_gps(*BOSTON_POINTS[0])
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    poly_4326 = shapely.geometry.Polygon(BOSTON_POINTS)
    poly_utm = shapely.geometry.Polygon([to_utm.transform(x, y) for x, y in BOSTON_POINTS])
    centre = poly_utm.envelope.centroid
    return types.SimpleNamespace(
        points=BOSTON_POINTS,
        polygon_4326=poly_4326,
        utm_crs=utm,
        center_x=centre.x,
        center_y=centre.y,
    )


# ---------------------------------------------------------------------------
# Fake LiDAR download (replaces scene_generation.USGS_LiDAR_HAG)
# ---------------------------------------------------------------------------
class FakeLidar:
    """
    Stand-in for ``generate_hag``. Set ``mode`` before the call:
      "ok"        writes test_hag.laz and test_hag.tif
      "laz_only"  writes test_hag.laz, then fails at the HAG raster step
                  (the crash that was originally reported)
      "no_cover"  raises like the real function when there is no LiDAR coverage
    """

    def __init__(self):
        self.mode = "ok"
        self.calls = []

    def generate_hag(self, polygon, data_dir, CRS="EPSG:3857", raise_on_error=False):
        self.calls.append({"data_dir": data_dir, "raise_on_error": raise_on_error})
        d = Path(data_dir)
        if self.mode == "no_cover":
            raise ValueError("No LiDAR data available for the selected region.")
        (d / "test_hag.laz").write_bytes(b"laz")
        if self.mode == "ok":
            (d / "test_hag.tif").write_bytes(b"tif")
            return
        msg = "writers.gdal: Unable to write empty raster"
        if raise_on_error:
            raise RuntimeError(msg)
        print(msg)


@pytest.fixture
def fake_lidar(monkeypatch):
    fake = FakeLidar()
    mod = types.ModuleType("scene_generation.USGS_LiDAR_HAG")
    mod.generate_hag = fake.generate_hag
    monkeypatch.setitem(sys.modules, "scene_generation.USGS_LiDAR_HAG", mod)
    return fake


# ---------------------------------------------------------------------------
# Fake terrain mesh builders (LiDAR mesh and DEM mesh)
# ---------------------------------------------------------------------------
class FakeMeshBuilder:
    def __init__(self):
        self.calls = []
        self.fail_with = None

    def __call__(self, src, ply_path, **kwargs):
        self.calls.append({"src": src, "ply": Path(ply_path), "kwargs": kwargs})
        if self.fail_with is not None:
            raise self.fail_with
        Path(ply_path).write_bytes(b"ply")


@pytest.fixture
def fake_lidar_mesh(monkeypatch):
    builder = FakeMeshBuilder()
    mod = types.ModuleType("scene_generation.lidar_terrain_mesh")
    mod.generate_terrain_mesh = builder
    monkeypatch.setitem(sys.modules, "scene_generation.lidar_terrain_mesh", mod)
    return builder


@pytest.fixture
def fake_dem_mesh(monkeypatch):
    from scene_generation import core

    builder = FakeMeshBuilder()
    monkeypatch.setattr(core, "generate_terrain_mesh_dem", builder)
    return builder


@pytest.fixture
def terrain_kwargs(tmp_path, scene_geometry):
    """Arguments for core.prepare_terrain_and_hag, minus terrain_mode / calibration."""
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    return dict(
        hag_tiff_path=None,
        data_dir=tmp_path,
        mesh_data_dir=mesh_dir,
        ground_polygon_4326=scene_geometry.polygon_4326,
        ground_scale=1.5,
        utm_crs=scene_geometry.utm_crs,
        center_x=scene_geometry.center_x,
        center_y=scene_geometry.center_y,
    )


# ---------------------------------------------------------------------------
# Small synthetic GeoTIFFs for the DEM tests
# ---------------------------------------------------------------------------
NODATA = -999999.0


def write_dem(path, array, crs, x0, y0, res=1.0, nodata=NODATA):
    """Write a single-band float64 GeoTIFF with its top-left corner at (x0, y0)."""
    with rasterio.open(
        path, "w", driver="GTiff", height=array.shape[0], width=array.shape[1],
        count=1, dtype="float64", crs=crs, transform=from_origin(x0, y0, res, res),
        nodata=nodata,
    ) as ds:
        ds.write(array.astype("float64"), 1)
    return path


@pytest.fixture
def sloped_dem(tmp_path):
    """
    200 m x 200 m DEM in UTM 19N (1 m cells) that rises 0.1 m per metre to the
    east, with a 20 m nodata strip on its east edge. Returns (path, info).
    """
    x0, y0 = 330000.0, 4690200.0
    dem = np.fromfunction(lambda r, c: 20 + 0.1 * c, (200, 200))
    dem[:, 180:] = NODATA
    path = write_dem(tmp_path / "sloped.tif", dem, "EPSG:32619", x0, y0)
    # scene centre placed exactly on the centre of cell (row 100, col 100)
    info = types.SimpleNamespace(x0=x0, y0=y0, cx=x0 + 100.5, cy=y0 - 100.5, z_centre=30.0)
    return path, info
