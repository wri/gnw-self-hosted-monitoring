# Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.
#
# For users who keep their own LOCAL copies of the zarrs (and the analysis script): lists
# which zarrs -- and whether the analysis script itself -- need to be (re-)copied, based on
# when each one's entry last changed in the manifest. The Landmark (indigenous/community)
# zarr is reported only if you can access the private folder that holds it (see the
# include_landmark flag below).
#
# When a zarr is updated, we never overwrite the existing zarr: a brand-new zarr is
# created and the manifest is updated atomically to point at it. The manifest records an
# "updatetime" for each zarr and for the analysis script, giving when that entry last
# changed.
#
# To install (same environment as post2020.py):
#  - Make sure python 3.12 or close to that is installed, and "pipenv" is installed
#  - Make a directory, copy check_updates.py into directory, cd to directory
#  - Run 'pipenv install xarray shapely pandas fiona zarr fsspec s3fs rioxarray "dask[array,dataframe,distributed,diagnostics]" coiled'
#
# Usage:
#    pipenv run python ./check_updates.py [-u] [date or date-time]
#
# Typical workflow:
#  - Run `check_updates.py` the first time (no ./lastchecktime yet): it lists all the zarr
#    names and current paths in the manifest -- the initial set of zarrs to copy to your
#    local bucket.
#  - When you are ready to copy them, run `check_updates.py -u`. This lists them again,
#    prints the current time (UTC), and records it in ./lastchecktime. Copy over all the
#    listed zarrs, update your local manifest.json, and point your local copy of the query
#    script at that manifest.
#  - Later, run `check_updates.py` any time to see only the zarrs whose paths changed since
#    the time in ./lastchecktime. When you are ready to bring those updates local, run
#    `check_updates.py -u` again to re-copy and advance ./lastchecktime to the current time.
#
# You can also supply a date or date-time on the command line (with or without -u) to use
# as the last check-and-copy time instead of ./lastchecktime -- useful if ./lastchecktime
# was accidentally removed or corrupted. All arguments are joined with spaces, so it need
# not be quoted, and many formats are accepted, e.g.:
#    pipenv run python ./check_updates.py 2026-06-15
#    pipenv run python ./check_updates.py -u 2026-06-15 13:45
#    pipenv run python ./check_updates.py June 15 2026
# If you give only a date (no time), the earliest time of that day (00:00) is assumed.
# A time with no timezone is interpreted as your local time.

import os
import sys
import json
from datetime import datetime, timezone

import fsspec
from dateutil import parser as dateparser

# Location of the global manifest that lists each zarr (name, S3 location, and the time
# that location was made the current value for the zarr).
manifest_uri = "s3://gnw-monitoring-data/post2020-manifest.json"

# Local file that records the last time the user checked for and copied over updates.
lastcheck_file = "./lastchecktime"

# Report the Landmark (indigenous/community) dataset only if this user can access the
# private folder that holds it (users are granted a role on it if allowed). To force the
# behavior, set include_landmark to True or False by hand in your local copy.
include_landmark = fsspec.filesystem("s3", requester_pays=True).exists("gnw-private-monitoring-data")


def parse_datetime(s: str) -> datetime:
    """Parse a date or date-time in many formats into a UTC-aware datetime.

    A date with no time defaults to the earliest time of day (00:00). A time with no
    timezone is interpreted as local time. Everything is converted to UTC so it can be
    compared against the manifest's "updatetime", which is recorded in UTC.
    """
    dt = dateparser.parse(s)
    # A naive datetime (no tzinfo) is presumed to be in the system's local timezone;
    # datetime.astimezone() treats a naive value as local time when converting to UTC.
    return dt.astimezone(timezone.utc)


def read_lastcheck_str() -> str | None:
    """Return the trimmed contents of ./lastchecktime, or None if missing/empty."""
    if not os.path.exists(lastcheck_file):
        return None
    with open(lastcheck_file) as f:
        content = f.read().strip()
    return content or None


def check_updates(manifest_uri: str, since: datetime) -> tuple[list[dict], dict | None]:
    """Return (zarr entries whose updatetime is after `since`, the analysis-script entry if
    its updatetime is after `since` else None)."""
    storage_options = {"requester_pays": True}
    with fsspec.open(manifest_uri, "r", **storage_options) as f:
        manifest = json.load(f)

    # Skip the Landmark dataset unless this user includes it (see include_landmark).
    updated = [entry for entry in manifest["zarrs"]
               if (include_landmark or entry["name"] != "indig_area_zarr")
               and parse_datetime(entry["updatetime"]) > since]

    # The analysis script is a top-level entry (location + updatetime), checked the same
    # way as a zarr. Older manifests may not have it, so tolerate its absence.
    script = manifest.get("analysis_script")
    script_updated = script if (script and parse_datetime(script["updatetime"]) > since) else None

    return updated, script_updated


def main() -> None:
    args = sys.argv[1:]
    update = "-u" in args
    # Anything that is not the -u flag is treated as the (space-joined) date/time.
    date_tokens = [a for a in args if a != "-u"]

    if date_tokens:
        # An explicit date/time on the command line overrides ./lastchecktime.
        source: str | None = " ".join(date_tokens)
        from_file = False
    else:
        source = read_lastcheck_str()
        from_file = True

    if source is None:
        # No date given and no ./lastchecktime yet: initial run, list every zarr.
        since = datetime.min.replace(tzinfo=timezone.utc)
        initial_run = True
    else:
        try:
            since = parse_datetime(source)
        except (ValueError, OverflowError) as e:
            where = f"{lastcheck_file} ('{source}')" if from_file else f"'{source}'"
            print(f"Error: could not parse {where} as a date/time: {e}")
            if from_file:
                print(f"Supply a date/time on the command line instead, or remove {lastcheck_file}.")
            sys.exit(1)
        initial_run = False

    updated, script = check_updates(manifest_uri, since)

    if not updated and script is None:
        print(f"No zarrs or the analysis script have been updated since {since.isoformat()}.")
    else:
        if updated:
            if initial_run:
                print("All zarrs and their current paths (the initial set to copy):")
            else:
                print(f"Zarrs updated since {since.isoformat()} (copy these again):")
            for entry in updated:
                print(f"  {entry['name']:<17}  {entry['location']:<62}  (updated {entry['updatetime']})")
        if script is not None:
            if updated:
                print("")
            if initial_run:
                print("Analysis script (copy this too):")
            else:
                print("The analysis script has been updated (copy it again):")
            print(f"  {script['location']}  (updated {script['updatetime']})")

    if update:
        # Record now (UTC): the user is bringing these updates local as of this time.
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with open(lastcheck_file, "w") as f:
            f.write(now + "\n")
        print(f"\nRecorded current time (UTC) in {lastcheck_file}: {now}")
        if updated or script is not None:
            print("Copy over everything listed above, update your local manifest.json,")
            print("and point your local copy of the query script at that manifest.")
    elif updated or script is not None:
        print("\nWhen you are ready to copy these over, re-run with -u to record the")
        print(f"current time in {lastcheck_file} (so later runs show only newer changes).")


if __name__ == "__main__":
    main()
