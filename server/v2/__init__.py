"""ZIP/LODES region previews. No legacy generation or Toy_ABM dependencies."""
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import requests

GEOGRAPHY_YEAR = 2020
LODES_YEAR = 2023
LODES_RELEASE = 'LODES8'
LODES_JOB_TYPE = 'JT01'
CACHE_DIR = Path(os.getenv('DELINEO_V2_CACHE_DIR', Path(__file__).resolve().parents[1] / 'data' / 'v2'))


def download_cached(url: str, destination: Path, progress_callback=None) -> Path:
    """Publish complete downloads atomically; concurrent readers never see partial files."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with requests.get(url, stream=True, timeout=(15, 180)) as response:
            response.raise_for_status()
            with NamedTemporaryFile(dir=destination.parent, delete=False) as output:
                temporary = Path(output.name)
                downloaded = 0
                last_report = time.monotonic()
                for chunk in response.iter_content(1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback is not None and time.monotonic() - last_report >= 10:
                        progress_callback(
                            f'Downloading {destination.name}: {downloaded / (1024 * 1024):.0f} MB'
                        )
                        last_report = time.monotonic()
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
