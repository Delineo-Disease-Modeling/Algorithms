#!/usr/bin/env python3
"""
Plot SafeGraph POI polygons from a monthly patterns CSV on an OSM map.

Examples:
  python scripts/plot_poi_polygons.py \
    --state OK --month 2019-01 --poi-cbg 401430069022 \
    --name-contains lafortune

  python scripts/plot_poi_polygons.py \
    --patterns-file data/OK/2019-01-OK.csv --poi-cbg 401430069022 \
    --placekey zzz-222@5r7-fqf-sbk --placekey 223-222@5r7-fqg-2kz --placekey zzw-222@5r7-fqf-sbk
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import folium
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping


SERVER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SERVER_DIR / "data"
OUTPUT_DIR = SERVER_DIR / "output"

if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

try:
    from geojsongen import load_state_shapefile  # type: ignore
except Exception:
    load_state_shapefile = None


STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06",
    "CO": "08", "CT": "09", "DE": "10", "DC": "11", "FL": "12",
    "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23",
    "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28",
    "MO": "29", "MT": "30", "NE": "31", "NV": "32", "NH": "33",
    "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38",
    "OH": "39", "OK": "40", "OR": "41", "PA": "42", "RI": "44",
    "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
    "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56", "PR": "72", "VI": "78", "GU": "66", "MP": "69",
    "AS": "60",
}

PALETTE = [
    "#dc2626", "#2563eb", "#16a34a", "#9333ea", "#ea580c",
    "#0891b2", "#4f46e5", "#65a30d", "#db2777", "#0f766e",
]


def normalize_cbg(raw) -> str:
    try:
        cbg = str(int(float(raw)))
    except (TypeError, ValueError):
        cbg = str(raw or "").strip()
    if len(cbg) == 11 and cbg.isdigit():
        cbg = cbg.zfill(12)
    return cbg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot POI polygons over OSM for a CBG/month")
    parser.add_argument("--patterns-file", help="Direct path to monthly patterns CSV")
    parser.add_argument("--state", help="Two-letter state code (e.g., OK)")
    parser.add_argument("--month", help="Month key YYYY-MM (e.g., 2019-01)")
    parser.add_argument("--poi-cbg", required=True, help="Target POI CBG (12-digit GEOID)")
    parser.add_argument("--name-contains", action="append", default=[], help="Case-insensitive name substring filter (repeatable)")
    parser.add_argument("--placekey", action="append", default=[], help="Exact placekey filter (repeatable)")
    parser.add_argument("--limit", type=int, default=50, help="Max rows to plot after filtering")
    parser.add_argument("--output", help="Output HTML file path")
    parser.add_argument("--no-cbg-boundary", action="store_true", help="Do not overlay the CBG polygon")
    return parser.parse_args()


def resolve_patterns_file(args: argparse.Namespace) -> Path:
    if args.patterns_file:
        p = Path(args.patterns_file)
        if not p.is_absolute():
            p = (SERVER_DIR / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Patterns file not found: {p}")
        return p

    if not args.state or not args.month:
        raise ValueError("Provide either --patterns-file or both --state and --month")

    state = str(args.state).upper()
    month = str(args.month)
    p = DATA_DIR / state / f"{month}-{state}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Monthly patterns file not found: {p}")
    return p


def load_matching_rows(patterns_file: Path, target_cbg: str, name_filters: List[str], placekeys: List[str], limit: int) -> pd.DataFrame:
    cols_needed = [
        "poi_cbg", "placekey", "location_name", "top_category", "sub_category",
        "street_address", "city", "region", "postal_code", "latitude", "longitude",
        "polygon_wkt", "polygon_class", "wkt_area_sq_meters", "raw_visit_counts", "raw_visitor_counts",
    ]
    results = []
    name_filters_l = [f.lower() for f in name_filters if str(f).strip()]
    placekey_set = {str(k).strip() for k in placekeys if str(k).strip()}

    for chunk in pd.read_csv(patterns_file, dtype=str, chunksize=10000, usecols=lambda c: c in cols_needed):
        if "poi_cbg" not in chunk.columns:
            continue

        chunk = chunk.copy()
        chunk["poi_cbg_num"] = pd.to_numeric(chunk["poi_cbg"], errors="coerce")
        chunk = chunk[chunk["poi_cbg_num"].notna()]
        if chunk.empty:
            continue
        chunk["poi_cbg_norm"] = chunk["poi_cbg_num"].astype("int64").astype(str).str.zfill(12)
        chunk = chunk[chunk["poi_cbg_norm"] == target_cbg]
        if chunk.empty:
            continue

        if placekey_set:
            chunk = chunk[chunk["placekey"].fillna("").isin(placekey_set)]
        if chunk.empty:
            continue

        if name_filters_l:
            names = chunk["location_name"].fillna("").str.lower()
            mask = False
            for sub in name_filters_l:
                mask = mask | names.str.contains(sub, regex=False)
            chunk = chunk[mask]
        if chunk.empty:
            continue

        results.append(chunk.drop(columns=["poi_cbg_num", "poi_cbg_norm"]))
        if limit > 0 and sum(len(df) for df in results) >= limit:
            break

    if not results:
        return pd.DataFrame()

    out = pd.concat(results, ignore_index=True)
    if limit > 0:
        out = out.head(limit)
    return out


def add_cbg_boundary(map_obj: folium.Map, poi_cbg: str) -> Optional[tuple]:
    if load_state_shapefile is None:
        return None
    state_fips = poi_cbg[:2]
    gdf = load_state_shapefile(state_fips)
    if gdf is None or len(gdf) == 0:
        return None
    if "GEOID" not in gdf.columns and "CensusBlockGroup" in gdf.columns:
        gdf = gdf.copy()
        gdf["GEOID"] = gdf["CensusBlockGroup"]
    if "GEOID" not in gdf.columns:
        return None
    gdf = gdf.copy()
    gdf["GEOID"] = gdf["GEOID"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(12)
    row = gdf[gdf["GEOID"] == poi_cbg]
    if row.empty:
        return None
    row = row.to_crs("EPSG:4326")
    geom = row.iloc[0].geometry
    boundary_group = folium.FeatureGroup(name=f"CBG boundary ({poi_cbg})", show=True)
    folium.GeoJson(
        data=mapping(geom),
        style_function=lambda _x: {
            "color": "#0f172a",
            "weight": 3,
            "fillColor": "#1d4ed8",
            "fillOpacity": 0.08,
        },
        tooltip=f"CBG {poi_cbg}",
    ).add_to(boundary_group)
    boundary_group.add_to(map_obj)
    return geom.bounds  # minx, miny, maxx, maxy


def safe_float(raw) -> Optional[float]:
    try:
        v = float(raw)
        if math.isnan(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def main() -> int:
    args = parse_args()
    poi_cbg = normalize_cbg(args.poi_cbg)
    patterns_file = resolve_patterns_file(args)

    rows = load_matching_rows(
        patterns_file=patterns_file,
        target_cbg=poi_cbg,
        name_filters=args.name_contains,
        placekeys=args.placekey,
        limit=args.limit,
    )
    if rows.empty:
        print(f"No matching rows found in {patterns_file} for poi_cbg={poi_cbg}")
        return 1

    # Terminal summary (easier than spreadsheet view)
    cols_to_show = [
        c for c in [
            "placekey", "location_name", "street_address", "city", "region", "postal_code",
            "latitude", "longitude", "polygon_class", "wkt_area_sq_meters",
            "top_category", "sub_category", "raw_visit_counts", "raw_visitor_counts"
        ] if c in rows.columns
    ]
    print("\nMatched rows:")
    print(rows[cols_to_show].to_string(index=False))

    # Map init center fallback: median of available lat/lon
    lat_vals = [v for v in (safe_float(x) for x in rows.get("latitude", [])) if v is not None]
    lon_vals = [v for v in (safe_float(x) for x in rows.get("longitude", [])) if v is not None]
    center_lat = sum(lat_vals) / len(lat_vals) if lat_vals else 36.0
    center_lon = sum(lon_vals) / len(lon_vals) if lon_vals else -95.0
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=15, tiles="OpenStreetMap")

    bounds_points: List[List[float]] = []
    if not args.no_cbg_boundary:
        cbg_bounds = add_cbg_boundary(fmap, poi_cbg)
        if cbg_bounds:
            minx, miny, maxx, maxy = cbg_bounds
            bounds_points.extend([[miny, minx], [maxy, maxx]])

    for idx, row in rows.reset_index(drop=True).iterrows():
        color = PALETTE[idx % len(PALETTE)]
        name = str(row.get("location_name") or "Unknown POI").strip()
        placekey = str(row.get("placekey") or "").strip()
        lat = safe_float(row.get("latitude"))
        lon = safe_float(row.get("longitude"))
        polygon_wkt = str(row.get("polygon_wkt") or "").strip()
        layer_label = f"#{idx+1} {name}"
        if placekey:
            layer_label = f"{layer_label} [{placekey}]"
        poi_group = folium.FeatureGroup(name=layer_label, show=True)

        popup_lines = [
            f"<b>#{idx+1} {name}</b>",
            f"placekey: {placekey or 'N/A'}",
            f"address: {str(row.get('street_address') or '').strip()}",
            f"city/state/zip: {str(row.get('city') or '').strip()} {str(row.get('region') or '').strip()} {str(row.get('postal_code') or '').strip()}",
            f"category: {str(row.get('top_category') or '').strip()}",
            f"subcat: {str(row.get('sub_category') or '').strip()}",
            f"lat/lon: {lat if lat is not None else 'N/A'}, {lon if lon is not None else 'N/A'}",
            f"polygon_class: {str(row.get('polygon_class') or '').strip() or 'N/A'}",
            f"wkt_area_sq_meters: {str(row.get('wkt_area_sq_meters') or '').strip() or 'N/A'}",
        ]
        popup_html = "<br>".join(popup_lines)

        # Add point marker
        if lat is not None and lon is not None:
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.95,
                popup=folium.Popup(popup_html, max_width=500),
                tooltip=f"#{idx+1} {name}",
            ).add_to(poi_group)
            bounds_points.append([lat, lon])

        # Add polygon if present
        if polygon_wkt:
            try:
                geom = shapely_wkt.loads(polygon_wkt)
                folium.GeoJson(
                    data=mapping(geom),
                    style_function=lambda _x, c=color: {
                        "color": c,
                        "weight": 2,
                        "fillColor": c,
                        "fillOpacity": 0.18,
                    },
                    tooltip=f"#{idx+1} {name}",
                    popup=folium.Popup(popup_html, max_width=500),
                ).add_to(poi_group)
                minx, miny, maxx, maxy = geom.bounds
                bounds_points.extend([[miny, minx], [maxy, maxx]])
            except Exception as e:
                print(f"Warning: failed to parse polygon for {placekey or name}: {e}")

        poi_group.add_to(fmap)

    if bounds_points:
        fmap.fit_bounds(bounds_points, padding=(20, 20))

    folium.LayerControl(collapsed=False).add_to(fmap)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = (SERVER_DIR / out_path).resolve()
    else:
        suffix = "-".join([f for f in args.name_contains[:2]]) if args.name_contains else "all"
        out_path = OUTPUT_DIR / f"poi-polygons-{poi_cbg}-{suffix}.html"
    fmap.save(str(out_path))
    print(f"\nSaved map: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
