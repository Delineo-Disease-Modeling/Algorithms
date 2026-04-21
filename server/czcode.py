from common_geo import build_cbg_centers, distance, get_neighboring_states, normalize_cbg as _normalize_cbg
from czcode_modules import (
    Clustering,
    Config,
    GraphBuilder,
    Helpers,
    cbg_population,
    setup_logging,
)
from czcode_modules.data_loading import DataLoader


def generate_cz(*args, **kwargs):
    from czcode_modules.pipeline import generate_cz as _generate_cz

    return _generate_cz(*args, **kwargs)


def main(*args, **kwargs):
    from czcode_modules.pipeline import main as _main

    return _main(*args, **kwargs)


if __name__ == "__main__":
    main()
