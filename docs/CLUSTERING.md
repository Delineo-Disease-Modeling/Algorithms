# Deep dive: how convenience zones are built

This page explains how Algorithms turns a seed location into a **convenience
zone**: the mobility graph it builds, the clustering algorithms, how they
differ, and which one to use. Read the [README](../README.md) first. For the
terms, see the
[Glossary](https://github.com/Delineo-Disease-Modeling/.github/blob/main/docs/GLOSSARY.md).

Everything here describes the code as of October 2026. If you change the
behavior, update this page in the same PR.

## What a zone is for

The simulator can only model the people inside the zone. Anyone who lives
outside it is invisible: they never visit, never infect anyone, and never get
infected. So a good zone keeps as much of its residents' real movement
**inside** the zone as possible, while staying small enough to simulate. Every
algorithm below makes that trade-off in a different way.

## Step 1: build the mobility graph

`czcode_modules/pipeline.py:generate_cz` does the setup that all the
algorithms share:

1. **Pick the states.** It uses the seed's state plus its neighbors, from
   `data/neighbor-states.json`. Zones near a state line can cross it.
2. **Load SafeGraph patterns** for those states and the requested month
   (`pattern_resolution.py` picks the file).
3. **Load CBG shapes** (TIGER shapefiles) and **CBG populations**
   (`data/cbg_b01.csv`).
4. **Build the graph** (`graph.py:GraphBuilder.gen_graph`):
   - each **node** is a CBG,
   - each **edge weight** is the number of visits between two CBGs. A row in
     the patterns data says "this POI, located in CBG B, got N visitors from
     CBG A" (`visitor_daytime_cbgs`), and that adds N to the A–B edge,
   - visits within a single CBG are stored on the node as `self_weight`.

   The graph is **undirected**: A→B and B→A visits are added together.
   `gen_digraph` builds a directed version for the algorithms that need one.
   Graphs are cached per patterns file and set of states (`cache_service.py`),
   so the second request for the same area is much faster.

## The key measure: containment

Most of the algorithms score a zone by its **containment**, which the code
calls the CZI (`metrics.py:calculate_movement_stats`):

```text
containment = movement inside the zone / (movement inside + movement crossing the boundary)
```

- "Movement inside" is every zone CBG's `self_weight`, plus the edges between
  two zone CBGs.
- "Crossing the boundary" is the edges from a zone CBG to a CBG outside the
  zone.

A containment of 0.8 means 80% of the visits that involve the zone stay inside
it. Higher is better, but it always goes up as the zone grows, so every
algorithm needs a stopping rule.

## The algorithms

All the algorithm names are listed in `VALID_ALGORITHMS`
(`algorithm_runner.py`). **The CZ Generator page only offers four options;**
the rest can only be used through the API.

| Name | In the UI as | Stops when | Idea |
| --- | --- | --- | --- |
| `mobility_prune` | **Mobility Prune (Recommended)** | Removing anything else would drop seed capture below the target (default 80%) | Start big, then trim |
| `greedy_fast` | Greedy Weight | Population ≥ `min_pop` | Add the neighbor with the most movement to the zone |
| `greedy_weight_seed_guard` | Greedy Weight + Seed Guard | Population ≥ `min_pop` | Greedy weight, but only CBGs near the seed attract more CBGs |
| `greedy_weight` | API only | Population ≥ `min_pop` | The same rule as `greedy_fast`, older implementation |
| `greedy_ratio` | API only | Population ≥ `min_pop` | Add the neighbor whose movement is most concentrated toward the zone |
| `czi_balanced` | API only (the API's default) | Population ≥ `min_pop` | Add the neighbor that gives the best containment, with a small distance penalty |
| `czi_optimal_cap` | API only | Solver finishes | Solve for the best containment under a population cap (optimization, not greedy) |
| `greedy_ttwa` | API only | Population ≥ `min_pop` and no candidate improves containment | Travel-to-work-area style, on the directed graph |

"Guided Connected Cities" in the UI isn't a server algorithm. See
[Guided Connected Cities](#guided-connected-cities) below.

### Mobility Prune (recommended)

`mobility_prune.py`. The module docstring has the full reasoning. The short
version:

1. **Start with every CBG that has direct movement to or from the seed**, and
   that has at least one resident. This captures all of the seed's movement.
2. **Trim.** Repeatedly remove the CBG with the **least seed movement per
   resident**, as long as **seed capture** stays at or above
   `mobility_prune_min_seed_capture` (default 0.80). Seed CBGs are never
   removed.

```text
seed capture = (seed self-visits + seed-to-seed visits + seed-to-member visits)
             / (seed self-visits + every visit touching a seed)
```

Some things to know about it:

- **It ignores `min_pop`.** The population target is recorded in the metadata
  (`legacy_min_population`) but has no effect on the result. The zone's size is
  set by the capture target and by how much movement the seed has. To get a
  smaller zone, lower the capture target.
- Each candidate's score depends only on its own edges to the seed, so one
  sorted pass gives the same answer as removing CBGs one at a time.
- Every remaining CBG touches the seed directly, so the zone can't break into
  disconnected pieces.
- It accepts **several seed CBGs** (`seed_cbgs`), for example a whole town.

### The greedy family

These all start from the seed and add one neighboring CBG at a time until the
zone's population reaches `min_pop`. They differ only in how they score
candidates:

- **`greedy_fast` / `greedy_weight`**: the candidate with the most movement to
  CBGs already in the zone. `greedy_fast` keeps a running list of neighbors, so
  it's quicker on large graphs.
- **`greedy_weight_seed_guard`**: like greedy weight, but only zone members
  within `seed_guard_distance_km` (default 20 km) of the seed **count toward
  a candidate's score**. A far-away CBG can still join, but it can't pull in
  even more distant CBGs. This stops the zone from drifting along a highway
  toward a bigger city.
- **`greedy_ratio`**: the candidate whose movement is most concentrated toward
  the zone (movement to the zone ÷ all its movement). This favors CBGs that are
  "about" the zone rather than just busy.
- **`czi_balanced`**: the candidate that gives the **highest containment after
  it's added**, minus a small distance penalty
  (`distance_penalty_weight` × `d / (d + distance_scale_km)`, with defaults of
  0.02 and 20 km).

Greedy algorithms can overshoot `min_pop` by up to one CBG's population, and
they stop early if they run out of neighbors.

### `czi_optimal_cap`

`optimal_czi.py`. Instead of growing step by step, it treats zone selection as
a **mixed-integer program** (using SciPy's MILP solver). It picks the set of
CBGs that maximizes containment, with the population between
`optimal_population_floor_ratio × min_pop` (default 90%) and `min_pop`. Note
that `min_pop` acts as a **cap** here, not a minimum. Only the
`optimal_candidate_limit` (default 120) CBGs most connected to the seed are
considered. Containment is a ratio, so the solver repeats with Dinkelbach
iterations (up to `optimal_max_iters`), each with a time limit. If SciPy's MILP
isn't available, it falls back to `czi_balanced`. It has no step-by-step trace.

### `greedy_ttwa`

`directed_algorithms.py`. It's based on "travel-to-work areas", a method used by
national statistics offices. It uses the **directed** graph and measures
containment two ways: the share of the zone's residents' trips that stay inside
(origin side), and the share of visits to the zone that come from inside
(destination side). It scores by the weaker of the two. It keeps adding the
best candidate until the population target is met **and** no candidate
improves containment. It logs a warning if the result is below
`containment_threshold` (0.70).

## Guided Connected Cities

This option is built in the browser (`Fullstack/.../use-generation-preview-submit.ts`)
from Algorithms' `POST /second-order-destinations` endpoint
(`server_app/second_order_destinations.py`). It's meant for a small town whose
residents mostly travel to *other* towns nearby. The endpoint:

1. finds the towns (grouped by ZIP code) that the seed's residents travel to,
2. for each town, picks a few **gateway CBGs** that capture most of that flow
   (up to 4 per town, aiming for 85% of the flow, at most 12,000 people per
   town),
3. marks up to 4 towns as **recommended**, keeping the total population under
   25,000.

The page starts with the seed plus the recommended towns' gateway CBGs. The
user can add or remove towns, and edit by hand, before finalizing. No
clustering algorithm runs.

## What happens after clustering

The browser shows the preview. The user can add or remove CBGs, using
`/frontier-candidates` (neighbors worth adding), `/candidate-pois`, and
`/cz-metrics` (live containment and population) to decide. Then
`/finalize-cz` creates the zone from the final list. The algorithm's output is
a starting point, not the final answer.

**Traces.** With `include_trace`, each algorithm records every step: which
candidates it considered, their scores, and which one it picked. The CZ
Generator's heat-map view shows this trace. Mobility Prune's trace shows the
trimming steps. `czi_optimal_cap` has no trace because it has no steps. To
keep the response small, traces can be delta-encoded and loaded separately
(`/clustering-trace/<id>`, `trace_encoding.py`).

## Adding or changing an algorithm

1. Implement it as a method on one of the mixins in `czcode_modules/`.
2. Register it in **three places**: `VALID_ALGORITHMS` and `_handlers()` in
   `algorithm_runner.py`, and `VALID_CLUSTER_ALGORITHMS` in
   `server_app/constants.py`. `test_algorithm_registry_consistency.py` fails if
   these disagree.
3. If it takes parameters, add them to the request parsing
   (`server_app/algorithm_params.py`) and to `pipeline.generate_cz`.
4. To offer it in the UI, add it to `CLUSTER_ALGORITHM_OPTIONS` and
   `CLUSTER_ALGORITHM_MANUAL` in `Fullstack/src/features/cz-generation/constants.ts`.
5. Compare it against the existing algorithms on real seeds with
   `server/scripts/compare_zone_algorithms.py`.

## Known limitations

- **Movement data is from SafeGraph's sample of devices**, not the whole
  population, and it's per month. The same seed can produce a different zone
  in a different month.
- **The graph ignores direction** for every algorithm except `greedy_ttwa`. A
  CBG that sends many visitors to the zone counts the same as one that receives
  many.
- **Zero-population CBGs** (parks, industrial areas) can hold POIs that
  residents visit, but Mobility Prune never adds them, because they have no
  residents to simulate. `greedy_fast` and `czi_balanced` skip them as well;
  `greedy_weight` can still add them.
- **The first request for a new state is slow.** It has to load patterns and
  shapefiles and build the graph. The ZIP code lookup library (`uszipcode`) may
  also download its database on first use.

## Where to start reading the code

1. `czcode_modules/pipeline.py`: `generate_cz()`
2. `czcode_modules/graph.py`: how movement becomes a graph
3. `czcode_modules/mobility_prune.py`: the recommended algorithm, with a long
   docstring
4. `czcode_modules/greedy_algorithms.py`: the greedy family
5. `server_app/analysis_service.py`: how `/cluster-cbgs` jobs run and report
   progress

Tests: `test_mobility_prune_algorithm.py`, `test_algorithm_runner.py`,
`test_graph_builder.py`, `test_frontier_candidates.py`,
`test_second_order_destinations.py`.
