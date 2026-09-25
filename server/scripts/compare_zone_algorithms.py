"""Compare convenience-zone algorithms on real seeds (size, capture, CZI, runtime, geography).

Run from Algorithms/server with an environment that has the server deps and geopandas:

    python scripts/compare_zone_algorithms.py --zip 74002 74103 74056 55901

Geography uses 2010-vintage CBGs to match SafeGraph patterns: centroids and land
area from the SafeGraph open-census metadata (nationwide), polygons from the
TIGER 2016 block-group shapefiles under data/shapefiles_2016 where present,
falling back to data/shapefiles/<ST>.geojson (2020 vintage; IDs may not match,
reported as ``nopoly``). CBGs without a polygon count as isolated components.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVER_DIR))
os.chdir(SERVER_DIR)

import geopandas as gpd  # noqa: E402
import networkx as nx  # noqa: E402
import pandas as pd  # noqa: E402
from shapely.geometry import MultiPoint, Point  # noqa: E402

from common_geo import STATE_ABBR_TO_FIPS, STATE_FIPS_TO_ABBR  # noqa: E402
from czcode_modules.clustering import Clustering  # noqa: E402
from czcode_modules.config import Config  # noqa: E402
from czcode_modules.data_loading import DataLoader  # noqa: E402
from czcode_modules.graph import GraphBuilder  # noqa: E402
from czcode_modules.metrics import Helpers, cbg_population  # noqa: E402
from czcode_modules.mobility_prune import seed_movement_accounting  # noqa: E402

LOG = logging.getLogger("compare_zone_algorithms")
ALBERS = "EPSG:5070"
FAR_KM = 100.0

_graphs: dict = {}
_polygons: dict = {}
_centroids = None


def load_graph(seed_cbg: str, start_date: datetime):
    config = Config(seed_cbg, 0, start_date=start_date)
    key = config.paths["patterns_csv"]
    if key not in _graphs:
        df = DataLoader(config, LOG).load_safegraph_data([])
        _graphs[key] = GraphBuilder(LOG).gen_graph(df)
    return config, _graphs[key]


def centroids():
    global _centroids
    if _centroids is None:
        meta = pd.read_csv(
            "data/safegraph_open_census_data_2016/metadata/cbg_geographic_data.csv",
            dtype={"census_block_group": str},
        ).set_index("census_block_group")
        points = gpd.GeoSeries(
            gpd.points_from_xy(meta.longitude, meta.latitude), index=meta.index, crs="EPSG:4326"
        ).to_crs(ALBERS)
        _centroids = (points, meta.amount_land)
    return _centroids


def polygons(state: str):
    if state not in _polygons:
        fips = STATE_ABBR_TO_FIPS.get(state)
        tiger = Path(f"data/shapefiles_2016/tl_2016_{fips}_bg/tl_2016_{fips}_bg.shp")
        fallback = Path(f"data/shapefiles/{state}.geojson")
        if tiger.exists():
            gdf = gpd.read_file(tiger)
            gdf["cbg"] = gdf.GEOID.astype(str).str.zfill(12)
        elif fallback.exists():
            gdf = gpd.read_file(fallback)
            gdf["cbg"] = gdf.CensusBlockGroup.astype(str).str.zfill(12)
        else:
            gdf = None
        _polygons[state] = (
            None if gdf is None else gdf.set_index("cbg").geometry.to_crs(ALBERS)
        )
    return _polygons[state]


def seed_capture(G, seeds, zone):
    total, always, by_cbg = seed_movement_accounting(G, set(seeds))
    captured = always + sum(w for cbg, w in by_cbg.items() if cbg in zone)
    return captured / total if total else 1.0


def geography(zone, seeds, pops):
    points, land = centroids()
    geoms = {}
    for cbg in zone:
        state = STATE_FIPS_TO_ABBR.get(cbg[:2])
        polys = polygons(state) if state else None
        if polys is not None and cbg in polys.index:
            geoms[cbg] = polys.loc[cbg]

    contiguity = nx.Graph()
    contiguity.add_nodes_from(zone)
    keys = list(geoms)
    index = gpd.GeoSeries([geoms[k] for k in keys]).sindex
    for key in keys:
        for j in index.query(geoms[key].buffer(50), predicate="intersects"):
            if keys[j] != key:
                contiguity.add_edge(key, keys[j])
    seed_component = set().union(
        *[nx.node_connected_component(contiguity, s) for s in seeds if s in contiguity]
    )

    seed_points = points.reindex(seeds).dropna()
    center = Point(seed_points.x.mean(), seed_points.y.mean())
    dist = {cbg: points[cbg].distance(center) / 1000 for cbg in zone if cbg in points.index}
    total_pop = sum(pops.values()) or 1
    far = [cbg for cbg, d in dist.items() if d > FAR_KM]
    near_points = [points[cbg] for cbg, d in dist.items() if d <= FAR_KM]
    return {
        "components": nx.number_connected_components(contiguity),
        "pct_res_contiguous_with_seed": 100 * sum(pops[c] for c in seed_component) / total_pop,
        "popw_mean_km": sum(dist[c] * pops[c] for c in dist) / total_pop,
        "median_km": float(pd.Series(dist).median()),
        "max_km": max(dist.values()),
        "far_cbgs": len(far),
        "far_residents": sum(pops[c] for c in far),
        "hull_km2": MultiPoint([points[c] for c in dist]).convex_hull.area / 1e6,
        "hull_within_100km_km2": MultiPoint(near_points).convex_hull.area / 1e6,
        "land_km2": float(land.reindex(list(zone)).sum()) / 1e6,
        "states": len({cbg[:2] for cbg in zone}),
        "nopoly": len(set(zone) - set(geoms)),
    }


def run(zip_code, seeds, algorithm, threshold, start_date, with_trace):
    config, G = load_graph(seeds[0], start_date)
    clustering = Clustering(config, LOG)
    fn = getattr(clustering, algorithm)
    started = time.perf_counter()
    zone, population, meta = fn(G, seeds, config.min_cluster_pop, min_seed_capture=threshold)
    runtime = time.perf_counter() - started
    row = {
        "zip": zip_code,
        "algorithm": algorithm,
        "cbgs": len(zone),
        "residents": int(population),
        "seed_capture": seed_capture(G, seeds, set(zone)),
        "czi": Helpers.calculate_movement_stats(G, set(zone))["ratio"],
        "runtime_s": runtime,
        "start_cbgs": meta.get("initial_cbg_count"),
        "start_residents": meta.get("initial_population"),
    }
    if with_trace:
        trace = []
        fn(G, seeds, config.min_cluster_pop, min_seed_capture=threshold, trace_collector=trace)
        row["trace_steps"] = len(trace)
        row["trace_mb"] = len(json.dumps(trace)) / 1e6
    pops = {cbg: max(0, int(cbg_population(cbg, config, LOG) or 0)) for cbg in zone}
    row.update(geography(zone, [s for s in seeds if s in G], pops))
    row["zone"] = sorted(zone)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--zip", nargs="+", default=["74002", "74103", "74056", "55901"])
    parser.add_argument("--algorithms", nargs="+", default=["mobility_prune"])
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--start-date", default="2021-04-01")
    parser.add_argument("--trace", action="store_true", help="also measure trace size (runs twice)")
    parser.add_argument("--out", help="write rows (with zone CBG lists) to this JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    zip_to_cbg = json.load(open("data/zip_to_cbg.json"))
    start_date = datetime.fromisoformat(args.start_date)
    rows = []
    for zip_code in args.zip:
        seeds = zip_to_cbg[zip_code]
        for algorithm in args.algorithms:
            row = run(zip_code, seeds, algorithm, args.threshold, start_date, args.trace)
            rows.append(row)
            print(
                f"{zip_code} {algorithm:15s} {row['cbgs']:4d} CBGs {row['residents']:8,d} res "
                f"capture {row['seed_capture']:.4f} CZI {row['czi']:.4f} {row['runtime_s']:6.2f}s | "
                f"comps {row['components']} contiguous {row['pct_res_contiguous_with_seed']:.1f}% "
                f"popw {row['popw_mean_km']:.1f} km median {row['median_km']:.1f} km "
                f">{FAR_KM:.0f}km {row['far_cbgs']} CBGs/{row['far_residents']:,d} res max {row['max_km']:.0f} km "
                f"hull {row['hull_km2']:,.0f} km2 (<={FAR_KM:.0f}km {row['hull_within_100km_km2']:,.0f}) "
                f"states {row['states']} nopoly {row['nopoly']}"
                + (f" | trace {row['trace_steps']} steps {row['trace_mb']:.1f} MB" if args.trace else ""),
                flush=True,
            )
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
