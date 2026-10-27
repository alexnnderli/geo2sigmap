from pathlib import Path
from pyproj import CRS
from .USGS1mLocator_parallel import *
from .dem_utils import *
from importlib_resources import files
from .dem_gpkg_data import get_fesm_paths
def generate_terrain_mesh_dem(
    aoi_poly,
    ply_save_path,
    utm_crs=None,
    center_x=None,
    center_y=None,
):
    """
    Build a terrain mesh from USGS 1 m DEM tiles. Needs no LiDAR data.

    Parameters
    ----------
    aoi_poly : shapely Polygon (EPSG:4326)
        Area to cover (normally the scaled scene polygon).
    ply_save_path : str | Path
        Output .ply path. ``dem.tif`` is written next to it.
    utm_crs : pyproj CRS | str | int, optional
        The scene's UTM CRS. Every DEM tile is reprojected into this one CRS so
        the mesh shares a coordinate frame with buildings and roads. If omitted,
        the UTM zone of the AOI centroid is used.
    center_x, center_y : float, optional
        Scene centre in ``utm_crs`` (the same values used to localize the
        buildings). When given, the mesh is translated so this point is the
        origin, and z is shifted so the ground at the centre is z = 0 —
        the same convention as the LiDAR terrain mesh. When omitted, the old
        behaviour (mesh centred on its own centroid) is kept.
    """
    ply_save_path = Path(ply_save_path)
    ply_save_path.parent.mkdir(parents=True, exist_ok=True)


    gpkg_files = get_fesm_paths()
    
    locator = USGS1mLocator(
        gpkg_files[1],
        gpkg_files[0],
        debug=True
    )
    locator.set_debug(False)
    links = locator.get_links(aoi_poly)
    if len(links) < 1:
        raise RuntimeError("No USGS 1 m DEM tiles cover this area.")

    target_epsg = CRS.from_user_input(utm_crs).to_epsg() if utm_crs is not None else None
 
    # Always mosaic into ONE zone. The old "multi_zone" mode (used whenever
    # there was more than one tile) reprojected each tile into its own zone,
    # which doesn't match the scene's single UTM zone.
    # overwrite=True: otherwise an existing dem.tif from an earlier run makes
    # the new mosaic land in dem_<uuid>.tif while the stale dem.tif is meshed.
    result = clip_reproject_dem_to_wgs84_utm(
        aoi_poly,
        links,
        mode="single_zone",
        target_epsg=target_epsg,
        mosaic=True,
        out_prefix=str(ply_save_path.parent / "dem"),
        overwrite=True,
    )
    dem_tif = result.get("path")
    if not dem_tif or not Path(dem_tif).exists():
        raise RuntimeError(
            "DEM clip/reproject produced no raster "
            f"(skipped tiles: {result.get('skipped_sources')})."
        )

    dem_to_ply(
        dem_tif,
        ply_save_path,
        stride=1,
        z_scale=1.0,
        center_x=center_x,
        center_y=center_y,
    )
    if not ply_save_path.exists():
        raise RuntimeError(f"DEM mesh was not written to {ply_save_path}.")
    return ply_save_path
    

