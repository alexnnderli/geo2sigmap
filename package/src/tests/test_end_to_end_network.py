"""
Real end-to-end runs that download USGS and Overture data. Slow; skipped
unless you pass --run-network:

    pytest --run-network -m network
"""
import numpy as np
import pytest

from conftest import BOSTON_POINTS

pytestmark = pytest.mark.network


def test_dem_plus_lidar_flags_build_dem_scene(tmp_path):
    """Main bug 1 on real data: both terrain flags give a DEM-based scene."""
    from scene_generation.core import Scene

    Scene()(BOSTON_POINTS, str(tmp_path), None, lidar_height_calibration=False,
            generate_building_map=False, lidar_terrain=True, dem_terrain=True)
    assert (tmp_path / "scene.xml").exists()
    assert (tmp_path / "mesh" / "dem_terrain.ply").exists()
    assert "mesh/dem_terrain.ply" in (tmp_path / "scene.xml").read_text()


def test_flat_scene_roads_are_lifted_off_the_ground(tmp_path):
    """Roads sit ROAD_Z_OFFSET_M above the flat ground to avoid z-fighting."""
    import open3d as o3d

    from scene_generation.core import ROAD_Z_OFFSET_M, Scene

    Scene()(BOSTON_POINTS, str(tmp_path), None, lidar_height_calibration=False,
            generate_building_map=False, generate_roads=True)
    roads = sorted((tmp_path / "mesh").glob("road_*.ply"))
    assert roads, "expected at least one road mesh in downtown Boston"
    z = np.asarray(o3d.t.io.read_triangle_mesh(str(roads[0])).vertex.positions.numpy())[:, 2]
    assert np.allclose(z, ROAD_Z_OFFSET_M)
