# Delineo Algorithms

The Algorithms service builds everything a simulation needs before it runs.
It groups census block groups into a **convenience zone**, generates a
**synthetic population** for the zone, and generates hour-by-hour **movement
patterns** for every person in it. It's a Flask server on port **1880**.

> **New to Delineo?** Read
> [Start Here](https://github.com/Delineo-Disease-Modeling/.github/blob/main/docs/START_HERE.md)
> first, and keep the
> [Glossary](https://github.com/Delineo-Disease-Modeling/.github/blob/main/docs/GLOSSARY.md)
> handy.

## What it does

1. **Clustering.** Starting from one or more seed CBGs, it builds a zone from
   SafeGraph foot-traffic data, so that most residents' movement stays inside
   it. There are several algorithms. The recommended one, Mobility Prune,
   trims down to a movement-capture target; most of the others grow up to a
   population target. See [`docs/CLUSTERING.md`](docs/CLUSTERING.md).
2. **Synthetic population.** It creates people, households, and places that
   match the zone's Census statistics (age, sex, household makeup). The output
   is called **papdata**.
3. **Movement patterns.** It decides where each person is at every timestep:
   home, work, school, or a POI. It uses each POI's real hourly popularity and
   dwell times.
4. **Zone tools for the UI.** Metrics, nearby candidate CBGs, POI lists, CBG
   shapes, and map exports for the CZ Generator page.

When a zone is finalized, Algorithms sends the papdata and patterns to
Fullstack, which stores them. Algorithms keeps no database of its own.

## Running it

Do the full setup in
[Setup](https://github.com/Delineo-Disease-Modeling/.github/blob/main/docs/SETUP.md)
first. In particular, you need Python 3.13, a `.venv` in this folder, and the
**`server/data/` folder, which is not in git.** Then:

```bash
cd server && ../.venv/bin/python server.py
```

This starts the Flask dev server on <http://localhost:1880>. In production,
Docker runs `server/waitress_app.py` instead.

### The data folder

Everything in `server/data/` comes from outside the repo:

- `patterns/<STATE>/<YYYY-MM>-<STATE>.parquet`: SafeGraph monthly patterns. A
  zone can only be built for a state and month that have a file here. The UI
  checks this with `GET /pattern-availability`.
- `shapefiles_2016/tl_2016_<FIPS>_bg/`: Census block-group shapes. Download them
  with `server/scripts/download_tiger_2016_bg.py --states MD OK`.
- `cbg_b01.csv`: population by age and sex for each CBG.
- `neighbor-states.json`, `zip_to_cbg.json`, and `TEST/test.csv`: lookup tables
  and a small test patterns file.

Population generation also makes live requests to the Census API, and to
OpenStreetMap to place homes, so it needs internet access.

### Environment variables

All are optional.

| Variable | Default | What it does |
| --- | --- | --- |
| `FULLSTACK_URL` | `http://localhost:3000` | Where to send new zones, papdata, and patterns |
| `CENSUS_API_KEY` | a shared key in `census_data.py` | Your own (free) Census API key |
| `DELINEO_PATTERNS_BINARY` | on | Send patterns in the binary DLNOPAT format. Set to `0` to send JSON. |
| `DELINEO_FRACTIONAL_DWELL` | on | Allow visits shorter than an hour |
| `DELINEO_PERF_TIMINGS` | off | Log how long each generation stage takes |

## Tests

From this folder (not `server/`):

```bash
.venv/bin/python -m pytest tests
```

The tests use small fixtures in `tests/algorithms_server/fixtures/` and don't
need the data folder.

## API

All routes are defined in `server/server_app/routes.py`. The browser calls
them directly from the CZ Generator page.

| Method | Route | What it does |
| --- | --- | --- |
| GET | `/seed-region?zip=` | Looks up the seed CBGs for a ZIP code |
| POST | `/cluster-cbgs` | Starts a clustering job and returns a `clustering_id` |
| GET | `/clustering-progress/<id>` | Streams the job's progress (SSE), ending with the result |
| GET | `/clustering-trace/<id>` | The step-by-step clustering trace, for the visualization |
| POST | `/finalize-cz` | Creates the zone from a final CBG list, then generates the population and patterns in the background |
| GET | `/generation-progress/<czone_id>` | Streams generation progress (SSE) |
| POST | `/cz-metrics` | Population and containment metrics for a candidate zone |
| POST | `/frontier-candidates` | Neighboring CBGs that could be added to a zone |
| POST | `/second-order-destinations` | Suggests other towns the seed's residents often travel to, which can be added as extra seeds ("connected cities") |
| POST | `/candidate-pois` | The POIs in a candidate CBG that draw the most visitors from the zone |
| GET | `/cbg-geojson` | CBG shapes as GeoJSON |
| GET | `/cbg-at-point` | The CBG that contains a latitude/longitude |
| GET | `/pattern-availability` | Which months of patterns data exist for a state |
| POST | `/export-cz-map-html` | Exports a zone map as a standalone HTML file |
| POST | `/generate-cz` | Older one-step version of cluster and finalize. The current UI doesn't use it. |

## Code map

```text
server/
├── server.py, waitress_app.py   # entry points (dev, production)
├── server_app/                  # the Flask app
│   ├── app_factory.py           #   builds the app, CORS, services
│   ├── routes.py                #   every HTTP route
│   ├── generation_service.py    #   finalize zone, then popgen + patterns + upload
│   ├── analysis_service.py      #   clustering jobs and zone-analysis endpoints
│   ├── fullstack_client.py      #   calls to Fullstack's /api
│   ├── pattern_resolution.py    #   picks the right patterns file for a state/month
│   └── constants.py             #   data paths, CORS origins, algorithm names, defaults
├── czcode.py, czcode_modules/   # clustering algorithms
├── popgen.py                    # synthetic population (gen_pop)
│   ├── cbg_demographics.py, census_data.py, residential.py, worker_assignment.py
│   └── papdata_convert.py       #   turns the population into papdata
├── patterns.py                  # movement patterns (gen_patterns)
│   ├── dwell.py, monthly_patterns.py, patterns_loader.py
│   └── patterns_codec.py        #   binary DLNOPAT encoder
├── geojsongen.py, common_geo.py # geography helpers
├── run_report.py                # per-run logs and summaries (written to server/output/)
└── scripts/                     # one-off tools: data download, benchmarks, plots
tests/algorithms_server/         # pytest suite
docs/                            # design documents
archive/                         # code from earlier semesters; not used by the server
```

## Things that trip people up

- **GEOIDs are 12-character strings.** Leading zeros matter, so never convert a
  GEOID to a number. `common_geo.normalize_cbg` repairs one that lost its zero.
- **There's no patterns file for my state.** Clustering and generation need
  data for the zone's state and month. Check `server/data/patterns/<STATE>/`.
- **Adding a clustering algorithm.** It has to be registered in three places:
  `VALID_ALGORITHMS` and the dispatch map in
  `czcode_modules/algorithm_runner.py`, and `VALID_CLUSTER_ALGORITHMS` in
  `server_app/constants.py`. `test_algorithm_registry_consistency.py` fails if
  they don't match.
- **Changing the papdata or patterns format.** Simulation and Fullstack read
  these too. Update the readers first, then the writer. See
  [How We Work](https://github.com/Delineo-Disease-Modeling/.github/blob/main/docs/HOW_WE_WORK.md#changes-that-touch-more-than-one-repo).

## Further reading

- [`docs/CLUSTERING.md`](docs/CLUSTERING.md): **start here.** How zones are
  built: the mobility graph, every clustering algorithm, and which one to use
- [`docs/POPULATION_AND_MOVEMENT.md`](docs/POPULATION_AND_MOVEMENT.md): how the
  synthetic population and hour-by-hour movement are generated, and the
  assumptions behind them
- [Data formats](https://github.com/Delineo-Disease-Modeling/.github/blob/main/docs/DATA_FORMATS.md):
  the exact shape of papdata and movement patterns
- [`docs/MOVEMENT_MODEL_REDESIGN.md`](docs/MOVEMENT_MODEL_REDESIGN.md): why POIs
  were overfilling, and how movement demand is now scaled
- [`server/MOVEMENT_FIX.md`](server/MOVEMENT_FIX.md): the fix for people staying
  home too much
- [`server/scripts/perf_benchmark.README.md`](server/scripts/perf_benchmark.README.md):
  benchmarking generation performance
