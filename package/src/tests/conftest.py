"""
Stub out heavy optional dependencies (open3d, pyvista, rasterio, pdal, ...)
so the terrain/HAG decision logic can be tested without a full GIS stack.
numpy, shapely, pyproj, geopandas and requests are real.
"""
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_STUBS = [
    "open3d", "open3d.core", "osmnx", "pyvista", "triangle", "tqdm",
    "PIL", "PIL.Image", "PIL.ImageDraw", "PIL.ImageChops",
    "rasterio", "rasterio.enums", "rasterio.warp", "rasterio.vrt", "rasterio.mask",
    "rasterio.merge", "rasterio.transform", "rasterio.windows",
    "duckdb", "trimesh", "laspy", "pdal", "plyfile",
    "scipy", "scipy.spatial", "scipy.constants", "scipy.interpolate",
    "importlib_resources", "pooch", "scene_generation.data",
]
for name in _STUBS:
    if name not in sys.modules:
        sys.modules[name] = MagicMock(name=name)
