# Delineo Algorithms

Algorithms provides the Flask service for convenience-zone analysis, synthetic
population generation, and movement generation. It uploads generated zone inputs
to the sibling Fullstack application's API.

Start with the shared [project overview](https://github.com/Delineo-Disease-Modeling/Fullstack/blob/main/docs/overview.md)
and [local walkthrough](https://github.com/Delineo-Disease-Modeling/Fullstack/blob/main/docs/getting-started.md).
The walkthrough includes a synthetic integration example and does not require
access to the private Deploy repository.

## Install and start this service

Use Python 3.12 for the native setup. From this repository's root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
cd server
FULLSTACK_URL=http://localhost:3000 ../.venv/bin/python server.py
```

The service listens on port **1880**. Start it from `server/` because some data
paths are relative to the current directory. Fullstack must be running for zone
creation and uploads. `FULLSTACK_URL` selects its origin.

Check the service without installing source data:

```bash
curl --fail 'http://localhost:1880/pattern-availability?state=OK&start_date=2021-04-05&end_date=2021-04-05'
```

An empty `available_months` list is expected in a fresh clone. There is no `/`
health route, so a 404 at `http://localhost:1880/` does not mean startup failed.

## Inputs for real zone generation

The public repository does not bundle the complete mobility and Census datasets.
The [data setup section](https://github.com/Delineo-Disease-Modeling/Fullstack/blob/main/docs/getting-started.md#7-prepare-data-for-a-real-convenience-zone)
describes access requirements and directory layout. In particular:

- Monthly mobility files go under `server/data/patterns/<STATE>/`, named
  `YYYY-MM-<STATE>.parquet` (also supported: `.csv.gz`, `.converted.csv`, `.csv`).
- Population data is read from `server/data/cbg_b01.csv`.
- CBG geometry uses 2016 TIGER/Line files under
  `server/data/shapefiles_2016/tl_2016_<FIPS>_bg/`.
- Census population generation also needs Census API connectivity; use your own
  `CENSUS_API_KEY` where needed.

Usable mobility statistics are required for movement generation. Missing inputs
raise an error; they do not trigger a home-only movement fallback. Selecting a
date without a matching monthly file also produces an error.

## Code map

| Location | Responsibility |
| --- | --- |
| `server/server.py` | Development server entry point |
| `server/server_app/` | App factory, routes, request parsing, progress jobs, Fullstack client |
| `server/czcode_modules/` | Mobility graphs, clustering algorithms, and metrics |
| `server/popgen.py` | Synthetic population and people/places bundle |
| `server/patterns.py` | Hourly movement generation; output keys are elapsed minutes |
| `server/patterns_loader.py` | Shared mobility file resolution and column selection |
| `tests/algorithms_server/` | Algorithm, data, route, and movement tests |
| `archive/` | Earlier implementations, outside the current startup path |

Zone and movement-generator durations are in hours. The Simulation API consumes
minutes, so a 24-hour zone supplies a 1,440-minute example run.

The [movement redesign notes](docs/MOVEMENT_MODEL_REDESIGN.md) include historical
proposals and later implementation updates. Use current code and the shared
overview to establish today's behavior.

## Contributing

`main` is deployed to production. Start changes from the latest `main` on a
short-lived branch. Coordinate cross-repository changes with the same branch
name and linked pull requests, and update the shared documentation when behavior
or setup changes.
