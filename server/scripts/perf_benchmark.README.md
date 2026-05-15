# Perf benchmark harness

End-to-end timing harness for the Delineo pipeline:
`generate_cz` -> `gen_pop` -> `gen_patterns` -> simulation -> `processSimulation`.

Used for the `ryad/perf-phase1` work. Drives all three repos in-process from
their `perf-phase1` worktrees (Algorithms, Simulation, Fullstack), records
per-stage timings, hashes of intermediate artifacts, and peak RSS, then writes
a structured `summary.json`.

## Usage

```
/opt/anaconda3/envs/delineo_env/bin/python perf_benchmark.py \
  2>&1 | tee <output_root>/full.log
```

Defaults: ZIP 74002, `mobility_prune` at `min_seed_capture=0.70`, `min_pop=5000`,
168h, start 2021-04-01, `DELINEO_BENCH_SEED=0`, 1 smoke + 3 measured.

## Environment variables

Set automatically by the harness (you can override):
- `DELINEO_PERF_TIMINGS=1` — turns on `[perf]` stage logs in all three repos.
- `DELINEO_BENCH_SEED=0` — deterministic seed for `gen_pop` and `ResidentialCache`.

Set these to point at non-default paths:
- `DELINEO_ROOT` — parent dir holding the Delineo sibling repos.
- `DELINEO_ALGORITHMS_WORKTREE`, `DELINEO_SIMULATION_WORKTREE`,
  `DELINEO_FULLSTACK_WORKTREE` — perf worktree paths.
- `DELINEO_ALGORITHMS_DATA_CWD` — directory the Algorithms code runs from
  (where `data/...` is resolved).
- `DELINEO_PATTERNS_FILE` — SafeGraph patterns parquet for the target month.
- `DELINEO_NODE_PROCESSOR` — path to `process_simulation.js`.
- `DELINEO_FULLSTACK_NODE_MODULES` — path to Fullstack's `node_modules`.
- `DELINEO_PERF_RUNS` — output directory (default: `$DELINEO_ROOT/_perf-runs`).

## CLI flags

- `--measured-runs N` (default 3)
- `--hours N` (default 168)
- `--min-pop N` (default 5000)
- `--min-seed-capture F` (default 0.70)
- `--start-date YYYY-MM-DD` (default 2021-04-01)
- `--bench-seed S` (default `"0"`; pass empty string to disable seeding)
- `--skip-smoke`
- `--output-root PATH`

## Output

- `<output_root>/<NN>-<label>/` — per-run artifacts (papdata, simdata,
  movement, map-cache).
- `<output_root>/summary.json` — `{env, args, runs[]}`. `runs[].hashes`
  include content hashes for `.gz` files (gunzipped) so gzip-header noise
  doesn't show as drift.
- `<output_root>/full.log` — full stdout of the last harness run.
