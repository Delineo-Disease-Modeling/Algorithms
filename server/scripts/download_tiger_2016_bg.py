#!/usr/bin/env python3
"""
Download and extract 2016 Census TIGER/Line block group shapefiles.

By default this fetches all 50 states plus DC from:
https://www2.census.gov/geo/tiger/TIGER2016/BG/

Each state's archive is extracted into:
Algorithms/server/data/shapefiles_2016/tl_2016_<FIPS>_bg/
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


BASE_URL = "https://www2.census.gov/geo/tiger/TIGER2016/BG"
DEFAULT_TIMEOUT_SEC = 120

# 50 states + DC. Territories are available via --include-territories.
STATE_FIPS_TO_ABBR = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

TERRITORY_FIPS_TO_ABBR = {
    "60": "AS",
    "66": "GU",
    "69": "MP",
    "72": "PR",
    "78": "VI",
}

REQUIRED_EXTENSIONS = (".shp", ".dbf", ".shx", ".prj")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_output = script_path.parents[1] / "data" / "shapefiles_2016"

    parser = argparse.ArgumentParser(
        description="Download and extract 2016 TIGER/Line block group shapefiles."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Destination folder (default: {default_output})",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=None,
        help="Subset of state abbreviations to fetch, e.g. MO KS OK",
    )
    parser.add_argument(
        "--include-territories",
        action="store_true",
        help="Also fetch PR, VI, GU, MP, and AS.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload and re-extract even if a state's files already exist.",
    )
    parser.add_argument(
        "--delete-zips",
        action="store_true",
        help="Delete each .zip archive after successful extraction.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT_SEC})",
    )
    return parser.parse_args()


def build_state_map(include_territories: bool) -> dict[str, str]:
    state_map = dict(STATE_FIPS_TO_ABBR)
    if include_territories:
        state_map.update(TERRITORY_FIPS_TO_ABBR)
    return state_map


def normalize_requested_states(
    requested_states: list[str] | None,
    state_map: dict[str, str],
) -> list[tuple[str, str]]:
    if not requested_states:
        return sorted(state_map.items(), key=lambda item: item[0])

    abbr_to_fips = {abbr: fips for fips, abbr in state_map.items()}
    normalized: list[tuple[str, str]] = []
    seen: set[str] = set()

    for raw_state in requested_states:
        abbr = str(raw_state).strip().upper()
        if not abbr:
            continue
        fips = abbr_to_fips.get(abbr)
        if not fips:
            valid = ", ".join(sorted(abbr_to_fips))
            raise ValueError(f"Unknown state abbreviation '{abbr}'. Valid values: {valid}")
        if fips in seen:
            continue
        seen.add(fips)
        normalized.append((fips, abbr))

    return normalized


def archive_name(state_fips: str) -> str:
    return f"tl_2016_{state_fips}_bg.zip"


def extract_dir_name(state_fips: str) -> str:
    return f"tl_2016_{state_fips}_bg"


def url_for_state(state_fips: str) -> str:
    return f"{BASE_URL}/{archive_name(state_fips)}"


def is_extract_complete(extract_dir: Path, state_fips: str) -> bool:
    if not extract_dir.is_dir():
        return False

    stem = f"tl_2016_{state_fips}_bg"
    return all((extract_dir / f"{stem}{ext}").exists() for ext in REQUIRED_EXTENSIONS)


def download_file(url: str, destination: Path, timeout: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=destination.name + ".",
        suffix=".part",
        delete=False,
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            with tmp_path.open("wb") as out_file:
                shutil.copyfileobj(response, out_file)
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def extract_zip(zip_path: Path, extract_dir: Path) -> None:
    clear_directory(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)


def process_state(
    state_fips: str,
    state_abbr: str,
    output_dir: Path,
    overwrite: bool,
    delete_zips: bool,
    timeout: int,
) -> str:
    zip_path = output_dir / archive_name(state_fips)
    extract_dir = output_dir / extract_dir_name(state_fips)
    url = url_for_state(state_fips)

    if not overwrite and is_extract_complete(extract_dir, state_fips):
        return f"[skip] {state_abbr} ({state_fips}) already present"

    if overwrite:
        clear_directory(extract_dir)
        if zip_path.exists():
            zip_path.unlink()

    if not zip_path.exists():
        print(f"[download] {state_abbr} ({state_fips}) <- {url}", flush=True)
        download_file(url, zip_path, timeout=timeout)
    else:
        print(f"[reuse] {state_abbr} ({state_fips}) zip already downloaded", flush=True)

    print(f"[extract] {state_abbr} ({state_fips}) -> {extract_dir}", flush=True)
    extract_zip(zip_path, extract_dir)

    if not is_extract_complete(extract_dir, state_fips):
        raise RuntimeError(
            f"Extraction for {state_abbr} ({state_fips}) did not produce the expected shapefile set"
        )

    if delete_zips:
        zip_path.unlink(missing_ok=True)
        return f"[done] {state_abbr} ({state_fips}) extracted; zip deleted"

    return f"[done] {state_abbr} ({state_fips}) extracted"


def main() -> int:
    args = parse_args()
    state_map = build_state_map(args.include_territories)

    try:
        states_to_fetch = normalize_requested_states(args.states, state_map)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Target directory: {args.output_dir}\n"
        f"States queued: {', '.join(abbr for _, abbr in states_to_fetch)}",
        flush=True,
    )

    failures: list[str] = []
    for state_fips, state_abbr in states_to_fetch:
        try:
            status = process_state(
                state_fips=state_fips,
                state_abbr=state_abbr,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
                delete_zips=args.delete_zips,
                timeout=args.timeout,
            )
            print(status, flush=True)
        except urllib.error.HTTPError as exc:
            failures.append(f"{state_abbr} ({state_fips}): HTTP {exc.code}")
            print(f"[error] {failures[-1]}", file=sys.stderr, flush=True)
        except urllib.error.URLError as exc:
            failures.append(f"{state_abbr} ({state_fips}): {exc.reason}")
            print(f"[error] {failures[-1]}", file=sys.stderr, flush=True)
        except Exception as exc:
            failures.append(f"{state_abbr} ({state_fips}): {exc}")
            print(f"[error] {failures[-1]}", file=sys.stderr, flush=True)

    if failures:
        print("\nFailures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nAll requested shapefiles are present.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
