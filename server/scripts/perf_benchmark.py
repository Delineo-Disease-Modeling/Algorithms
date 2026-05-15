from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


# Paths can be overridden via env vars so this script is portable across machines.
# Defaults match the parent directory holding the Delineo sibling repos and the
# `ryad/perf-phase1` worktrees under `_worktrees/`.
ROOT = Path(os.environ.get("DELINEO_ROOT", "/Users/ryad/Code/delineo"))
ALGORITHMS_WORKTREE = Path(
    os.environ.get("DELINEO_ALGORITHMS_WORKTREE", ROOT / "_worktrees" / "algorithms-perf-phase1")
)
SIMULATION_WORKTREE = Path(
    os.environ.get("DELINEO_SIMULATION_WORKTREE", ROOT / "_worktrees" / "simulation-perf-phase1")
)
FULLSTACK_WORKTREE = Path(
    os.environ.get("DELINEO_FULLSTACK_WORKTREE", ROOT / "_worktrees" / "fullstack-perf-phase1")
)
ALGORITHMS_DATA_CWD = Path(
    os.environ.get("DELINEO_ALGORITHMS_DATA_CWD", ROOT / "Algorithms" / "server")
)
PATTERNS_FILE = Path(
    os.environ.get(
        "DELINEO_PATTERNS_FILE",
        ROOT
        / "Algorithms"
        / "server"
        / "data"
        / "gdrive_1qtMYMT0sHQ7IYojSstbYtqUyTGJNukk1"
        / "data"
        / "patterns"
        / "OK"
        / "2021-04-OK.parquet",
    )
)
NODE_PROCESSOR = Path(
    os.environ.get(
        "DELINEO_NODE_PROCESSOR",
        Path(__file__).resolve().parent / "process_simulation.js",
    )
)

ZIP_CODE = "74002"
SEED_CBGS = ["401139400081", "401139400082", "401139400083"]
START_DATE = datetime(2021, 4, 1)
DEFAULT_HOURS = 168
DEFAULT_MIN_POP = 5000
DEFAULT_MIN_SEED_CAPTURE = 0.70


def configure_imports() -> None:
    sys.path.insert(0, str(SIMULATION_WORKTREE))
    sys.path.insert(0, str(ALGORITHMS_WORKTREE / "server"))
    os.chdir(ALGORITHMS_DATA_CWD)


def _git_head(repo: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _node_version() -> str:
    try:
        out = subprocess.run(
            ["node", "--version"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def collect_env_metadata() -> dict[str, Any]:
    return {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "node_version": _node_version(),
        "uname": platform.platform(),
        "git_head": {
            "algorithms": _git_head(ALGORITHMS_WORKTREE),
            "simulation": _git_head(SIMULATION_WORKTREE),
            "fullstack": _git_head(FULLSTACK_WORKTREE),
        },
    }


def _maxrss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024.0 / 1024.0 if sys.platform == "darwin" else rss / 1024.0


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    is_gzip = path.suffix == ".gz"
    opener = gzip.open if is_gzip else open
    with opener(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def timed(label: str, func):
    started_at = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - started_at
    print(f"[perf] {label}: {elapsed:.3f}s", flush=True)
    return result, elapsed


def write_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(value, f, separators=(",", ":"))


def run_node_processor(
    run_dir: Path,
    simdata_path: Path,
    movement_path: Path,
    papdata_id: str,
    length_minutes: int,
    run_number: int,
) -> tuple[Path, float]:
    map_cache_path = run_dir / "map-cache.json"
    env = os.environ.copy()
    env["DB_FOLDER"] = str(run_dir / "db") + "/"
    env["DELINEO_PERF_TIMINGS"] = "1"
    env["NODE_PATH"] = str(ROOT / "Fullstack" / "node_modules")

    started_at = time.perf_counter()
    completed = subprocess.run(
        [
            "node",
            str(NODE_PROCESSOR),
            "--simdata",
            str(simdata_path),
            "--patterns",
            str(movement_path),
            "--papdata-id",
            papdata_id,
            "--map-cache",
            str(map_cache_path),
            "--length",
            str(length_minutes),
            "--sim-data-id",
            str(run_number),
        ],
        cwd=str(FULLSTACK_WORKTREE),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    elapsed = time.perf_counter() - started_at
    print(completed.stdout, end="", flush=True)
    print(f"[perf] processSimulation subprocess total: {elapsed:.3f}s", flush=True)
    return map_cache_path, elapsed


def run_once(label: str, index: int, output_root: Path, args) -> dict[str, Any]:
    from czcode import generate_cz
    from patterns import gen_patterns
    from patterns_loader import PatternsData
    from popgen import gen_pop
    from simulator.runner import SimulationRunner

    run_dir = output_root / f"{index:02d}-{label}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    print(
        f"[bench] run={label} zip={ZIP_CODE} algorithm=mobility_prune "
        f"min_seed_capture={args.min_seed_capture:.2f} min_pop={args.min_pop} "
        f"start={args.start_date} hours={args.hours} bench_seed={os.environ.get('DELINEO_BENCH_SEED', '')}",
        flush=True,
    )

    rss_before_mb = _maxrss_mb()

    geoids_result, generate_cz_seconds = timed(
        "generate_cz",
        lambda: generate_cz(
            SEED_CBGS[0],
            args.min_pop,
            patterns_file=str(PATTERNS_FILE),
            start_date=START_DATE,
            algorithm="mobility_prune",
            seed_cbgs=SEED_CBGS,
            mobility_prune_min_seed_capture=args.min_seed_capture,
        ),
    )
    geoids, _map_obj, gdf = geoids_result
    cbg_set = set(geoids.keys())

    shared_data, patterns_load_seconds = timed(
        "patterns load",
        lambda: PatternsData.load([str(PATTERNS_FILE)], cbg_set=cbg_set),
    )
    papdata, gen_pop_seconds = timed(
        "gen_pop",
        lambda: gen_pop(geoids, gdf=gdf, shared_data=shared_data),
    )
    patterns, gen_patterns_seconds = timed(
        "gen_patterns",
        lambda: gen_patterns(papdata, START_DATE, args.hours, shared_data=shared_data),
    )

    papdata_id = "papdata"
    write_gzip_json(run_dir / "db" / f"{papdata_id}.gz", papdata)

    length_minutes = args.hours * 60
    simdata = {
        "czone_id": index,
        "length": length_minutes,
        "randseed": False,
        "initial_infected_count": 1,
        "disease_name": "COVID-19",
        "variants": ["Delta"],
        "dmp_mode": "off",
        "model_path_by_variant": {"Delta": None},
        "interventions": [
            {
                "time": 0,
                "mask": 0.4,
                "vaccine": 0.2,
                "capacity": 1.0,
                "lockdown": 0.0,
                "selfiso": 0.5,
            }
        ],
    }

    def data_loader(_url, timeout=360):
        return papdata, patterns

    simulation_output_dir = run_dir / "simulation-output"
    simulator_logger = logging.getLogger("simulator.runner")
    previous_simulator_log_level = simulator_logger.level
    simulator_logger.setLevel(logging.WARNING)
    try:
        simulation_result, simulation_seconds = timed(
            "simulation total",
            lambda: SimulationRunner(
                simdata,
                enable_logging=False,
                output_dir=str(simulation_output_dir),
                data_loader=data_loader,
            ).run(),
        )
    finally:
        simulator_logger.setLevel(previous_simulator_log_level)
    if "error" in simulation_result:
        raise RuntimeError(f"Simulation failed: {simulation_result['error']}")

    simdata_path = Path(simulation_result["simdata"])
    movement_path = Path(simulation_result["patterns"])
    map_cache_path, process_seconds = run_node_processor(
        run_dir,
        simdata_path,
        movement_path,
        papdata_id,
        length_minutes,
        index,
    )

    rss_after_mb = _maxrss_mb()
    result = {
        "label": label,
        "index": index,
        "zip": ZIP_CODE,
        "seed_cbgs": SEED_CBGS,
        "algorithm": "mobility_prune",
        "min_seed_capture": args.min_seed_capture,
        "min_pop": args.min_pop,
        "bench_seed": os.environ.get("DELINEO_BENCH_SEED", ""),
        "start_date": args.start_date,
        "hours": args.hours,
        "length_minutes": length_minutes,
        "cbg_count": len(geoids),
        "population": int(sum(geoids.values())),
        "people_count": len(papdata.get("people", {})),
        "home_count": len(papdata.get("homes", {})),
        "place_count": len(papdata.get("places", {})),
        "pattern_timesteps": len(patterns),
        "rss_mb": {
            "before": rss_before_mb,
            "after": rss_after_mb,
            "delta": rss_after_mb - rss_before_mb,
        },
        "timings": {
            "generate_cz": generate_cz_seconds,
            "patterns_load": patterns_load_seconds,
            "gen_pop": gen_pop_seconds,
            "gen_patterns": gen_patterns_seconds,
            "simulation": simulation_seconds,
            "process_simulation": process_seconds,
        },
        "hashes": {
            "geoids": stable_hash(geoids),
            "papdata": stable_hash(papdata),
            "patterns": stable_hash(patterns),
            "simdata_file": file_hash(simdata_path),
            "movement_file": file_hash(movement_path),
            "map_cache_file": file_hash(map_cache_path),
        },
        "artifacts": {
            "run_dir": str(run_dir),
            "simdata": str(simdata_path),
            "movement": str(movement_path),
            "map_cache": str(map_cache_path),
        },
    }
    print("[bench-result] " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument(
        "--output-root",
        default=os.environ.get("DELINEO_PERF_RUNS", str(ROOT / "_perf-runs")),
    )
    parser.add_argument("--hours", type=int, default=DEFAULT_HOURS)
    parser.add_argument("--min-pop", type=int, default=DEFAULT_MIN_POP)
    parser.add_argument("--min-seed-capture", type=float, default=DEFAULT_MIN_SEED_CAPTURE)
    parser.add_argument("--start-date", default=START_DATE.date().isoformat())
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--bench-seed", default="0",
                        help="Value for DELINEO_BENCH_SEED; pass empty string to disable.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    os.environ["DELINEO_PERF_TIMINGS"] = "1"
    if args.bench_seed:
        os.environ["DELINEO_BENCH_SEED"] = args.bench_seed
    else:
        os.environ.pop("DELINEO_BENCH_SEED", None)

    configure_imports()
    if not PATTERNS_FILE.exists():
        raise FileNotFoundError(PATTERNS_FILE)
    if not NODE_PROCESSOR.exists():
        raise FileNotFoundError(NODE_PROCESSOR)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env_metadata = collect_env_metadata()
    print(f"[bench-env] {json.dumps(env_metadata, sort_keys=True)}", flush=True)

    results = []
    run_index = 0
    if not args.skip_smoke:
        results.append(run_once("smoke", run_index, output_root, args))
        run_index += 1

    for measured_index in range(args.measured_runs):
        results.append(run_once(f"measured-{measured_index + 1}", run_index, output_root, args))
        run_index += 1

    summary = {
        "env": env_metadata,
        "args": {
            "measured_runs": args.measured_runs,
            "hours": args.hours,
            "min_pop": args.min_pop,
            "min_seed_capture": args.min_seed_capture,
            "start_date": args.start_date,
            "bench_seed": args.bench_seed,
            "skip_smoke": args.skip_smoke,
        },
        "runs": results,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[bench-summary] {summary_path}", flush=True)


if __name__ == "__main__":
    main()
