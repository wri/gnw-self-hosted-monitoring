# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
#
# To install:
#  - Make sure python 3.12 or close to that is installed
#  - Make sure "pipenv" is installed
#  - Make a directory, copy ghg.py into directory, cd to directory
#  - Run 'pipenv install xarray shapely pandas fiona zarr fsspec s3fs rioxarray "dask[array,dataframe,distributed,diagnostics]" coiled'
#
# Now you can run:
#    pipenv run python ./ghg.py <geojson path | shapefile path> <year, in 2020-2025>
#
# The geometry input is always a file, because every geometry must supply its own agricultural
# yield. For a shapefile, you must unzip the set of files (if needed) and then specify the path
# to the .shp file.
#
# This computes a DLUC (direct land-use change) greenhouse-gas emissions-factor analysis for
# each input geometry: the 20-year discounted CO2/CH4/N2O emissions divided by the geometry's
# agricultural production. Production (Mg) = total_area (ha) * yield (kg/ha) / 1000, so an
# emissions factor ef_gas is in MgCO2e per Mg of production. The per-geometry yield comes from a
# "yield" property on each geojson/shapefile feature (kg/ha); every feature must have a positive
# yield or the run stops.

import sys
import json
from datetime import datetime
import warnings

import xarray as xr
import numpy as np
import pandas as pd
import fiona
import rioxarray  # noqa: F401 — needed for .rio accessor

from shapely.geometry import shape

import dask
import concurrent.futures

import fsspec
import s3fs

# Single shared filesystem used for opening all the zarrs. The buckets holding the
# zarrs are in us-east-1, so pinning the region avoids S3 redirect round-trips on
# every request.
s3fs_filesystem = s3fs.S3FileSystem(client_kwargs={"region_name": "us-east-1"}, requester_pays=True)

# Location of the manifest that lists each zarr (name, S3 location, description, and
# the time that location was last updated). If you make your own local copy of the
# zarrs, you can change this to point to your customized version of the manifest on
# either an S3 or local filesystem.
manifest_uri = "s3://gnw-monitoring-data/ghg-manifest.json"

# The zarr entries this script requires from the manifest, in the order used for the
# header. The manifest may also contain other entries -- newer ones added for later
# script versions, or older ones kept for backward compatibility -- which this script
# ignores.
required_zarrs = [
    "pixel_area_zarr",
    "filtered_tcl_zarr",
    "forest_gross_emissions_co2",
    "forest_gross_emissions_ch4",
    "forest_gross_emissions_n2o",
]

# The minimum manifest version this script needs. The manifest's top-level "version" is
# bumped whenever the manifest changes in any way. This is NOT checked on every run -- it
# is only reported (as an extra hint) when a required zarr is missing, to explain that an
# outdated manifest is the likely cause.
min_manifest_version = 1

def load_manifest(manifest_uri: str) -> dict:
    # Supply requester_pays option only for S3 paths.
    storage_options = {"requester_pays": True} if manifest_uri.startswith("s3://") else {}
    with fsspec.open(manifest_uri, "r", **storage_options) as f:
        manifest = json.load(f)
    return manifest

def open_single_dataset(name: str, uri: str) -> tuple[str, xr.Dataset]:
    ds = xr.open_zarr(s3fs_filesystem.get_mapper(uri))
    ds.rio.write_crs("EPSG:4326", inplace=True)
    return name, ds

def open_datasets(zarr_uris: dict[str, str], required_names: list[str],
                  manifest_version: int) -> dict[str, xr.Dataset]:
    # Only open the zarrs this script version needs (required_names)
    missing = [name for name in required_names if name not in zarr_uris]
    present = [name for name in required_names if name in zarr_uris]

    datasets: dict[str, xr.Dataset] = {}
    open_errors: dict[str, str] = {}

    # Use a ThreadPoolExecutor to fire off all S3 requests simultaneously.
    if present:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(present)) as executor:
            future_to_name = {
                executor.submit(open_single_dataset, name, zarr_uris[name]): name
                for name in present
            }
            for future in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    name, ds = future.result()
                    datasets[name] = ds
                except Exception as e:
                    open_errors[name] = str(e)

    # If anything the script needs is missing or failed to open, report it all at once and
    # stop.
    if missing or open_errors:
        lines = ["ERROR: could not load all the zarrs this script requires."]
        if missing:
            lines += [
                "",
                "Required zarrs missing from the manifest:",
                *[f"  {name}" for name in missing],
                "",
                f"The manifest at {manifest_uri} does not list every zarr this version of",
                "the script needs. You are most likely running a newer version of the script",
                "against an outdated manifest -- if you keep local copies of the zarrs and",
                "manifest, re-copy the latest manifest and any new zarrs it references.",
            ]
            # Only when a required zarr is missing do we bother checking the manifest
            # version, to add a hint about how out-of-date the manifest is.
            if manifest_version < min_manifest_version:
                lines += [
                    "",
                    f"This script needs manifest version {min_manifest_version} or later, but the",
                    f"manifest is version {manifest_version}.",
                ]
        if open_errors:
            lines += [
                "",
                "Required zarrs listed in the manifest but which could not be opened:",
                *[f"  {name} ({zarr_uris[name]}): {err}" for name, err in open_errors.items()],
            ]
        sys.exit("\n".join(lines))

    return datasets


def print_header(descriptions: list[str], year: int) -> None:
    print("")
    print("Data versions:")
    for description in descriptions:
        # Indent each description by two spaces, and any continuation lines (after an
        # embedded newline) by four.
        print("  " + description.replace("\n", "\n    "))
    print("\nComputing 20-year discounted GHG emission factors (MgCO2e per Mg production)")
    print(f"for year {year}, using emissions from {year - 19}-{year}")
    print("")
    print("Analysis start time: ", datetime.now())
    print("")


# Units shown on a second header line under each column (blank for columns without one, e.g.
# "name").
COLUMN_UNITS = {
    "total area": "(ha)",
    "yield": "(kg/ha)",
    "production": "(Mg)",
    "ef_co2": "(MgCO2e/Mg)",
    "ef_ch4": "(MgCO2e/Mg)",
    "ef_n2o": "(MgCO2e/Mg)",
}


def _format_cell(v) -> str:
    """Render one output cell as a string, matching the 4-dp float style used elsewhere."""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def format_with_units(df: pd.DataFrame, separator: bool = False) -> str:
    """Render df as a right-justified table like df.to_string(index=False), but with a units
    line directly under the column names. Every cell is pre-formatted to a string so the units
    row can be inserted as an ordinary row without disturbing the numeric formatting. With
    separator=True, a full-width rule of '=' is drawn between the header/units and the data."""
    disp = df.map(_format_cell)
    units_row = {col: COLUMN_UNITS.get(col, "") for col in disp.columns}
    disp = pd.concat([pd.DataFrame([units_row]), disp], ignore_index=True)
    text = disp.to_string(index=False)
    if separator:
        # Line 0 is the header, line 1 the units; insert a '=' rule before the data rows.
        lines = text.split("\n")
        lines.insert(2, "=" * max(len(line) for line in lines))
        text = "\n".join(lines)
    return text


def process_file(input_path: str, datasets: dict[str, xr.Dataset], descriptions: list[str], year: int) -> pd.DataFrame:
    try:
        with fiona.open(input_path, 'r') as source:
            print(f"Successfully opened file: {input_path}")
            print(f"Driver used: {source.driver}")
            features = list(source)
    except fiona.errors.DriverError:
        print(f"Error: Could not open file at '{input_path}'. It may not exist or is corrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

    def feat_name(f):
        return f.properties.get("Location_Name") or f.properties.get("Location_N") or f.id

    def feat_yield(f):
        y = f.properties.get("yield")
        return float(y) if pd.notna(y) else None

    if len(features) == 0:
        print(f"No features in {input_path}, aborting")
        sys.exit(1)

    # Every feature must have a positive yield (needed to compute emissions factors, else
    # they would be NaN and could be stored unnoticed). Scan all features first, list any
    # with a missing or non-positive yield, and stop before doing any analysis.
    bad = []
    for feature in features:
        y = feat_yield(feature)
        if y is None or y <= 0:
            bad.append((feat_name(feature), y))
    if bad:
        print(f"Error: {len(bad)} of {len(features)} geometries in {input_path} have a missing or "
              f"non-positive yield (a positive yield is required to compute emissions factors):")
        for name, y in bad:
            print(f"  {name}: yield={'missing' if y is None else y}")
        sys.exit(1)

    print_header(descriptions, year)
    dl = []
    for feature in features:
        name = feat_name(feature)
        print("Feature:", name)
        r = process_geojson(feature, datasets=datasets, year=year, crop_yield=feat_yield(feature), name=name)
        print(r.to_string(index=False), "\n")
        dl.append(r)

    print("")
    return pd.concat(dl)


def clip_to_geojson(ds: xr.Dataset, geojson: dict) -> xr.DataArray | None:
    """Clip a layer to the geometry, returning its band_data DataArray, or None if the layer
    has no data over the geometry (an empty bounding-box selection, or no valid pixels within
    the polygon).

    A None result means different things depending on the layer. For a dense layer that covers
    the whole world (pixel area, the pre-filtered TCL), None means the AOI is too small / off
    the grid -- the caller checks pixel_area once for that. For a sparse layer (the emissions
    layers, whose grids do not cover every part of the world), None just means the layer is
    absent over this geometry (no emissions of that gas here, so it contributes nothing).
    """
    geom = shape(geojson)
    sliced = ds.sel(
        x=slice(geom.bounds[0], geom.bounds[2]),
        y=slice(geom.bounds[3], geom.bounds[1]),
    ).squeeze("band")
    # An empty selection means the layer's grid does not reach this area at all.
    if sliced["band_data"].size == 0:
        return None
    try:
        return sliced.rio.clip([geom]).band_data
    except rioxarray.exceptions.NoDataInBounds:
        # Non-empty selection but no valid pixels within the polygon.
        return None


def process_geojson(geojson: dict, datasets: dict[str, xr.Dataset], year: int,
                    crop_yield: float | None = None, name=None) -> pd.DataFrame:
    # Clip the pixel-area layer first and use it as the single "AOI too small" check. It is a
    # dense, global layer, so if it has no data over the geometry the AOI really is too small
    # (or off the grid)
    pixel_area = clip_to_geojson(datasets["pixel_area_zarr"], geojson)
    if pixel_area is None:
        where = f" for feature '{name}'" if name is not None else ""
        sys.exit(f"Error: AOI is too small{where}. Please select a larger AOI.")
    pixel_area = pixel_area.astype(np.float64)

    # `filtered_tcl_zarr` is the pre-filtered TCL layer: it holds the loss year (1-25
    # for 2001-2025) only where the loss qualifies for GHG counting and 0 everywhere
    # else.
    loss_year = clip_to_geojson(datasets["filtered_tcl_zarr"], geojson)
    assert loss_year is not None    # dense layer: present wherever pixel_area is (checked above)

    # per-gas emissions * pixel area. Each emissions layer is in Mg CO2e per ha, so multiplying
    # by the pixel area (m^2) and dividing the final sum by 10000 (m^2 -> ha) yields tonnes CO2e
    # (and turns "total area" into hectares). An absent (sparse) emissions layer contributes
    # nothing.
    base = {}
    for gas, zarr_name in (("co2", "forest_gross_emissions_co2"),
                           ("ch4", "forest_gross_emissions_ch4"),
                           ("n2o", "forest_gross_emissions_n2o")):
        emissions = clip_to_geojson(datasets[zarr_name], geojson)
        base[gas] = xr.zeros_like(pixel_area) if emissions is None \
            else emissions.astype(np.float64) * pixel_area

    # Apply the target year's 20-year discounted emissions. A pixel lost in `year` gets weight
    # 0.0975, and each year older gets 0.005 less, down to 0.0025 for a loss 19 years earlier;
    # losses outside that window get weight 0. All the gas variables are summed together (in
    # parallel) in the single dask compute below.
    offset = (year - 2000) - loss_year    # 0 for a loss in `year`, up to 19 for `year`-19
    weight = xr.where((loss_year >= 1) & (offset >= 0) & (offset <= 19),
                      0.0975 - 0.005 * offset, 0.0)
    variables = {"total area": pixel_area}
    for gas in ("co2", "ch4", "n2o"):
        variables[gas] = base[gas] * weight

    ds = xr.Dataset(variables)

    results_dask: dask.dataframe.DataFrame = (
        ds.sum(dim=("x", "y"))
        .to_dask_dataframe()
        .drop(["spatial_ref", "band"], axis=1)
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
        row = (results_dask.compute() / 10000).iloc[0]

    # Convert the discounted emissions to emissions factors. production (Mg) = total_area (ha)
    # * yield (kg/ha) / 1000; ef_gas (MgCO2e per Mg of production) = emissions_gas / production.
    total_area = float(row["total area"])
    production = total_area * crop_yield / 1000.0 if crop_yield is not None else float("nan")
    divisor = production if production > 0 else float("nan")   # avoid divide-by-zero -> NaN

    out: dict[str, float] = {
        "total area": round(total_area, 4),
        "yield": round(crop_yield, 4) if crop_yield is not None else float("nan"),
        "production": round(production, 4)}
    for gas in ("co2", "ch4", "n2o"):
        out[f"ef_{gas}"] = round(float(row[gas]) / divisor, 4)

    df = pd.DataFrame([out])
    if name is not None:
        df.insert(loc=0, column="name", value=name)
    return df


def main() -> None:
    pd.set_option('display.float_format', '{:.4f}'.format)

    # First argument is the geometry input file (geojson/shapefile); the second is the target
    # year (e.g. 2025). A file is required because every geometry must carry its own
    # agricultural yield.
    if len(sys.argv) < 3:
        print("Usage: python ./ghg.py <geojson | shapefile> "
              "<year, in 2020-2025>")
        sys.exit(1)
    geom_arg = sys.argv[1]
    year_arg = sys.argv[2]
    try:
        year = int(year_arg)
    except ValueError:
        sys.exit(f"Error: the year must be a single year (e.g. 2025), got '{year_arg}'.")
    if not 2020 <= year <= 2025:
        sys.exit(f"Error: the year must be within 2020-2025 (got '{year_arg}').")

    print("Opening zarrs")
    manifest = load_manifest(manifest_uri)
    manifest_version = manifest.get("version", 0)
    manifest_by_name = {entry["name"]: entry for entry in manifest["zarrs"]}
    zarr_uris = {name: entry["location"] for name, entry in manifest_by_name.items()}
    datasets = open_datasets(zarr_uris, required_zarrs, manifest_version)
    analysis_start = datetime.now()
    # Header lists each required zarr's description (in required order), except pixel_area.
    descriptions = [manifest_by_name[name]["description"]
                    for name in required_zarrs if name != "pixel_area_zarr"]

    print(format_with_units(process_file(geom_arg, datasets, descriptions, year), separator=True))

    analysis_end = datetime.now()
    print("\nAnalysis end time: ", analysis_end)
    print(f"Total analysis time (s): {(analysis_end - analysis_start).total_seconds():.1f}")


if __name__ == "__main__":
    try:
        main()
    finally:
        # s3fs sometimes emits a harmless "Unclosed client session / connector" message as
        # python is shutting down. Silence the loop's exception handler so the exit is quiet
        # on every path -- normal completion, the "AOI too small" error, or the no-yield exit.
        s3fs_filesystem.loop.set_exception_handler(lambda loop, context: None)
