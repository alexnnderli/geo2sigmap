# Tests for the terrain / roads fixes

Pytest suite covering the changes described in *overture_merge error fixes*.
Everything runs offline in a few seconds: LiDAR, DEM, Overture and OSM downloads
are replaced with fakes (see `conftest.py`).

## Run

From the repo, with your normal Geo2SigMap environment active and the package
installed (`pip install -e .`):

```
pip install pytest
pytest tests                      # offline suite
pytest tests --run-network        # also the 2 real-download tests (slow)
```

PDAL and laspy are **not** needed for the offline suite.

## What each file covers

| File | Fix it checks |
|---|---|
| `test_flag_resolution.py` | Main bugs 1 & 2: which terrain source is used; roads skipped with a warning |
| `test_prepare_terrain.py` | Main bug 1: only fetch what each feature needs; DEM survives a failed HAG raster; LiDAR failures raise with the real error; HAG path returned (known issue #2) |
| `test_scene_flags.py` | The same fixes through `Scene()` itself, so a run actually uses the new logic; scene.xml points at the right terrain mesh; OSM height calibration gets the HAG raster |
| `test_generate_hag.py` | `generate_hag(raise_on_error=True)` re-raises the PDAL error and fails if test_hag.laz / .tif were not written |
| `test_dem_utils.py` | Downstream 1: DEM mesh centred on the scene centre (x = y = z = 0), pixel centres, missing cells don't shift it; tiles from another UTM zone reprojected into the scene zone; one dem.tif on re-runs; `debug=True` doesn't recurse |
| `test_dem_pipeline.py` | `generate_terrain_mesh_dem` passes the scene CRS and centre through and fails loudly |
| `test_cli.py` | `run_scene` turns a RuntimeError into a logged error and exit code 1 |
| `test_end_to_end_network.py` | Real data: DEM + LiDAR flags give a DEM scene; roads lifted 0.1 m (skipped by default) |
