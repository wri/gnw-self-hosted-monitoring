# To install:
#  - Make sure python 3.12 or close to that is installed
#  - Make sure "pipenv" is installed
#  - Make a directory, copy post2020.py into directory, cd to directory
#  - Run 'pipenv install xarray shapely pandas fiona zarr fsspec s3fs rioxarray "dask[array,dataframe,distributed,diagnostics]" coiled'
#
# Now you can run:
#    pipenv run python ./post2020.py [geojson file path | shapefile path | GADM id (e.g. IDN.24.5)]
#
# For a shapefile, you must unzip the set of files (if needed) and then specify the
# path to the .shp file. If you specify a GADM id (e.g. IDN.24.5 or BRA.6.8), then it
# looks up the geometry using https://data-api.globalforestwatch.org. To use this option, you
# also need curl and jq installed, and you must do 'export API_KEY={your GFW api key}'
#
# The Landmark (indigenous/community) results are included automatically if your AWS
# credentials can access the private folder that holds that layer; otherwise they are
# omitted. See the include_landmark flag below to force it on or off.

import os
import sys
import subprocess
import json
import re
from datetime import datetime
import warnings

import xarray as xr
import numpy as np
import pandas as pd
import fiona
import rioxarray  # noqa: F401 — needed for .rio accessor

from shapely import wkb
from shapely.geometry import mapping, shape

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
manifest_uri = "s3://gnw-monitoring-data/post2020-manifest.json"

# Include the Landmark (indigenous/community) layer in the analysis only if this user can
# access the private folder that holds it. Users are granted a role on that folder if they
# are allowed to use the Landmark data. To force the behavior, set include_landmark to
# True or False by hand in your local copy of this script.
include_landmark = s3fs_filesystem.exists("gnw-private-monitoring-data")

# The zarr entries this script requires from the manifest, in the order used for the
# header. The manifest may also contain other entries -- newer ones added for later
# script versions, or older ones kept for backward compatibility -- which this script
# ignores.
required_zarrs = [
    "pixel_area_zarr",
    "tcl_zarr",
    "sbtn_area_zarr",
    "jrc_area_zarr",
    *(["indig_area_zarr"] if include_landmark else []),
    "int_dist_zarr",
    "tcl_drivers_zarr",
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


def print_header(descriptions: list[str]) -> None:
    print("")
    print("Data versions:")
    for description in descriptions:
        # Indent each description by two spaces, and any continuation lines (after an
        # embedded newline) by four.
        print("  " + description.replace("\n", "\n    "))
    print("")
    print("Analysis start time: ", datetime.now())
    print("")


def create_gadm_cmd(api_key, gid) -> str:
    cmd = f"curl --header Content-Type:application/json --header x-api-key:{api_key} https://data-api.globalforestwatch.org/dataset/gadm_administrative_boundaries/v4.1.85/query/json -d "

    if re.match("^[A-Z][A-Z][A-Z]\\.[0-9]+\\.[0-9]+", gid):
        obj = "'{" + f'"sql": "SELECT gfw_geojson FROM data WHERE gid_2 like %27{gid}%%%27 AND adm_level = %272%27"' + "}'"
    else:
        obj = "'{" + f'"sql": "SELECT gfw_geojson FROM data WHERE gid_1 like %27{gid}%%%27 AND adm_level = %271%27"' + "}'"

    cmd += obj + ' | jq -r ".data[0].gfw_geojson"'
    return cmd


def wkb_to_geojson_feature(wkbstr: str) -> dict:
    geom = wkb.loads(bytes.fromhex(wkbstr))
    subobj = mapping(geom)
    return subobj


def process_file(input_path: str, datasets: dict[str, xr.Dataset], descriptions: list[str]) -> pd.DataFrame:
    try:
        with fiona.open(input_path, 'r') as source:
            print(f"Successfully opened file: {input_path}")
            print(f"Driver used: {source.driver}")

            print_header(descriptions)
            dl = []
            for feature in source:
                name = feature.properties.get("Location_Name") or feature.properties.get("Location_N") or feature.id
                print("Feature:", name)
                r = process_geojson(feature, datasets=datasets, name=name)
                print(r.to_string(index=False), "\n")
                dl.append(r)

            print("")
            return pd.concat(dl)

    except fiona.errors.DriverError:
        print(f"Error: Could not open file at '{input_path}'. It may not exist or is corrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def clip_ds_to_geojson(ds: xr.Dataset, geojson: dict) -> xr.Dataset:
    geom = shape(geojson)
    sliced = ds.sel(
        x=slice(geom.bounds[0], geom.bounds[2]),
        y=slice(geom.bounds[3], geom.bounds[1]),
    ).squeeze("band")

    try:
        clipped = sliced.rio.clip([geom])
    except rioxarray.exceptions.NoDataInBounds:
        raise ValueError("AOI is too small. Please select a larger AOI ")

    return clipped


def process_geojson(geojson: dict, datasets: dict[str, xr.Dataset], name=None) -> pd.DataFrame:
    pixel_area = clip_ds_to_geojson(datasets["pixel_area_zarr"], geojson).astype(np.float64)
    sbtn_area = clip_ds_to_geojson(datasets["sbtn_area_zarr"], geojson).astype(np.float64)
    jrc_area = clip_ds_to_geojson(datasets["jrc_area_zarr"], geojson).astype(np.float64)
    intdist_zarr = clip_ds_to_geojson(datasets["int_dist_zarr"], geojson)
    tcl_drivers_zarr = clip_ds_to_geojson(datasets["tcl_drivers_zarr"], geojson)
    tcl_zarr = clip_ds_to_geojson(datasets["tcl_zarr"], geojson)

    # 4019 days from 2014/12/31 to 2026/1/1
    # Confidence encoding is 2 (nominal), 3 (high), and 4 (highest)
    alert_area = ((intdist_zarr.alert_date >= 4019) * (intdist_zarr.confidence >= 3) * pixel_area)
    # Important to do (sbtn_area/jrc_area > 0) rather than "!= 0" to process Nans in
    # area datasets correctly, since (Nan > 0) is false, but (Nan != 0) is true.
    sbtn_alert_area = alert_area * (sbtn_area > 0)
    jrc_alert_area = alert_area * (jrc_area > 0)

    variables = {"total area": pixel_area.band_data,
                 "sbtn_area": sbtn_area.band_data,
                 "sbtn_loss_area": (tcl_zarr.band_data > 20) * sbtn_area.band_data,
                 "jrc_area": jrc_area.band_data,
                 "jrc_loss_area": (tcl_zarr.band_data > 20) * jrc_area.band_data}

    # The Landmark (indigenous/community) column is included only when accessible (see
    # include_landmark), kept in the same position as before.
    if include_landmark:
        indig_area = clip_ds_to_geojson(datasets["indig_area_zarr"], geojson).astype(np.float64)
        variables["indig_area"] = indig_area.band_data

    variables.update({"alert_area": alert_area.band_data,
                      "sbtn_alert_area": sbtn_alert_area.band_data,
                      "jrc_alert_area": jrc_alert_area.band_data,
                      "perm_agric_area": (tcl_drivers_zarr.band_data == 1) * (pixel_area.band_data),
                      "hard_commod_area": (tcl_drivers_zarr.band_data == 2) * (pixel_area.band_data),
                      "shift_cult_area": (tcl_drivers_zarr.band_data == 3) * (pixel_area.band_data),
                      "logging_area": (tcl_drivers_zarr.band_data == 4) * (pixel_area.band_data),
                      "wildfire_area": (tcl_drivers_zarr.band_data == 5) * (pixel_area.band_data),
                      "settle_infra_area": (tcl_drivers_zarr.band_data == 6) * (pixel_area.band_data),
                      "other_nat_area": (tcl_drivers_zarr.band_data == 7) * (pixel_area.band_data)})

    ds = xr.Dataset(variables)

    results_dask: dask.dataframe.DataFrame = (
        ds.sum(dim=("x", "y"))
        .to_dask_dataframe()
        .drop(["spatial_ref", "band"], axis=1)
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in cast", category=RuntimeWarning)
        results_df: pd.DataFrame = (results_dask.compute() / 10000).round(4)

    if name is not None:
        results_df.insert(loc=0, column="name", value=name)
    return results_df


def main() -> None:
    pd.set_option('display.float_format', '{:.4f}'.format)
    print("Opening zarrs")
    manifest = load_manifest(manifest_uri)
    manifest_version = manifest.get("version", 0)
    manifest_by_name = {entry["name"]: entry for entry in manifest["zarrs"]}
    zarr_uris = {name: entry["location"] for name, entry in manifest_by_name.items()}
    datasets = open_datasets(zarr_uris, required_zarrs, manifest_version)
    # Header lists each required zarr's description (in required order), except pixel_area.
    descriptions = [manifest_by_name[name]["description"]
                    for name in required_zarrs if name != "pixel_area_zarr"]
    if len(sys.argv) > 1:
        if re.match("^010", sys.argv[1]):
            # Handle a WKB string
            print_header(descriptions)
            geojson = wkb_to_geojson_feature(sys.argv[1])
            print(process_geojson(geojson, datasets=datasets).to_string(index=False))
        elif re.match("^[A-Z][A-Z][A-Z]\\.", sys.argv[1]):
            # Handle a GADM id string, e.g. "IDN.4.4"
            gid = sys.argv[1]
            cmd = create_gadm_cmd(os.environ['API_KEY'], gid)
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, shell=True)
            print_header(descriptions)
            geojson = json.loads(result.stdout.strip())
            print(process_geojson(geojson, datasets=datasets).to_string(index=False))
        else:
            # Handle a geometry file
            print(process_file(sys.argv[1], datasets, descriptions).to_string(index=False))
    else:
        # Some default test cases, if no arg provided.
        # IDN, forest_extent = 994.6, nf_loss = 59.95, area = 1006
        wkbstr = "01030000000100000005000000e7aed1e4156e5840d583f1ef0d6706405123d124216e5840bb5dcfdae43506403dbcd0243d70584057ee7202b43806406a50d1e404705840f75140ece2700640e7aed1e4156e5840d583f1ef0d670640"
        # BRA, forest_extent = 8504788, nf_loss = 96996, area = 8850617
        # wkbstr = "01030000000100000005000000fa7637c15bb94ec0e082b62ea03811c0fa7637c15bb94ec010b38ccd0d7a19c070b278d981f94cc010b38ccd0d7a19c070b278d981f94cc0e082b62ea03811c0fa7637c15bb94ec0e082b62ea03811c0"
        print_header(descriptions)
        geojson = wkb_to_geojson_feature(wkbstr)
        print(process_geojson(geojson, datasets=datasets).to_string(index=False))

    print("\nAnalysis end time: ", datetime.now())

    # s3fs sometimes emits a harmless "Unclosed client session / connector" message
    # as python is shutting down. Silence the exception handler so that the exit is
    # quiet when the analysis was successful.
    s3fs_filesystem.loop.set_exception_handler(lambda loop, context: None)


if __name__ == "__main__":
    main()
