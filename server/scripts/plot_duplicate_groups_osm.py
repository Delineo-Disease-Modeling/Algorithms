#!/usr/bin/env python3
"""
Generate one OSM map per duplicate POI cluster.

Inputs:
- cluster_members.csv (from group_same_location_pois.py)
- one or more monthly patterns CSV files containing polygon_wkt

Outputs:
- one HTML map per group
- an index HTML linking all generated maps
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import folium
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import mapping


SERVER_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = SERVER_DIR / "data"
DEFAULT_CLUSTER_MEMBERS = (
    DATA_DIR / "integrity_reports" / "same_location_clusters" / "cluster_members.csv"
)
DEFAULT_OUTPUT_DIR = SERVER_DIR / "output" / "duplicate_group_maps"

PALETTE = [
    "#dc2626",
    "#2563eb",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#65a30d",
    "#db2777",
    "#0f766e",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot duplicate POI groups on OSM with polygon overlays."
    )
    parser.add_argument(
        "--cluster-members",
        default=str(DEFAULT_CLUSTER_MEMBERS),
        help=f"Path to cluster_members.csv (default: {DEFAULT_CLUSTER_MEMBERS})",
    )
    parser.add_argument(
        "--patterns-file",
        action="append",
        default=[],
        help="Direct patterns CSV path(s). Repeat for multiple files.",
    )
    parser.add_argument(
        "--patterns-glob",
        action="append",
        default=["data/OK/*.csv"],
        help="Glob(s) for patterns CSVs when --patterns-file not provided. Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for per-group maps (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--group-id",
        action="append",
        default=[],
        help="Only include specific group_id values (repeatable).",
    )
    parser.add_argument(
        "--name-filter",
        default=None,
        help="Only include groups where a member name contains this text.",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        help="Only include groups containing member rows in this region/state code (repeatable).",
    )
    parser.add_argument(
        "--min-members",
        type=int,
        default=2,
        help="Minimum members required in a group (default: 2).",
    )
    parser.add_argument(
        "--min-members-with-pattern",
        type=int,
        default=2,
        help="Minimum members with matching rows in patterns data to emit a map (default: 2).",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="Limit number of output groups. 0 means no limit.",
    )
    return parser.parse_args()


def _safe_float(raw: object) -> Optional[float]:
    try:
        value = float(raw)
        if math.isnan(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def _norm(text: object) -> str:
    return str(text or "").strip().lower()


def _slug(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    if not slug:
        return "group"
    return slug[:max_len]


def _resolve_user_path(path_text: str) -> Path:
    p = Path(path_text).expanduser()
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def _resolve_server_path(path_text: str) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return (SERVER_DIR / p).resolve()


def _read_cluster_members(path: Path) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            group_id = str(row.get("group_id") or "").strip()
            placekey = str(row.get("placekey") or "").strip()
            if not group_id or not placekey:
                continue
            groups[group_id].append(row)
    return groups


def _resolve_patterns_files(args: argparse.Namespace) -> List[Path]:
    if args.patterns_file:
        files = [_resolve_user_path(p) for p in args.patterns_file]
    else:
        files: List[Path] = []
        for pat in args.patterns_glob:
            candidates = [_resolve_user_path(pat), _resolve_server_path(pat)]
            seen: set[Path] = set()
            for full_pat in candidates:
                if full_pat in seen:
                    continue
                seen.add(full_pat)
                # Keep glob support for both cwd-relative and SERVER_DIR-relative patterns.
                if "*" in str(full_pat) or "?" in str(full_pat) or "[" in str(full_pat):
                    if full_pat.parent.exists():
                        files.extend(sorted(full_pat.parent.glob(full_pat.name)))
                else:
                    files.append(full_pat)

    existing = [p for p in files if p.exists()]
    return sorted(set(existing))


def _filter_groups(
    groups: Dict[str, List[dict]],
    min_members: int,
    group_ids: Iterable[str],
    name_filter: Optional[str],
    regions: Iterable[str],
) -> Dict[str, List[dict]]:
    allowed = {str(g).strip() for g in group_ids if str(g).strip()}
    name_filter_norm = _norm(name_filter)
    region_set = {str(r).strip().upper() for r in regions if str(r).strip()}

    out: Dict[str, List[dict]] = {}
    for gid, rows in groups.items():
        if len(rows) < min_members:
            continue
        if allowed and gid not in allowed:
            continue
        if name_filter_norm:
            if not any(name_filter_norm in _norm(r.get("location_name")) for r in rows):
                continue
        if region_set:
            row_regions = {str(r.get("region") or "").strip().upper() for r in rows}
            if row_regions.isdisjoint(region_set):
                continue
        out[gid] = rows
    return out


def _score_pattern_row(row: dict) -> Tuple[int, int]:
    polygon = str(row.get("polygon_wkt") or "").strip()
    lat = _safe_float(row.get("latitude"))
    lon = _safe_float(row.get("longitude"))
    return (1 if polygon else 0, 1 if (lat is not None and lon is not None) else 0)


def _load_pattern_lookup(
    patterns_files: List[Path],
    needed_placekeys: set[str],
) -> Dict[str, dict]:
    lookup: Dict[str, dict] = {}
    if not patterns_files or not needed_placekeys:
        return lookup

    cols = [
        "placekey",
        "poi_cbg",
        "location_name",
        "street_address",
        "city",
        "region",
        "postal_code",
        "latitude",
        "longitude",
        "polygon_wkt",
        "polygon_class",
        "wkt_area_sq_meters",
        "top_category",
        "sub_category",
        "raw_visit_counts",
        "raw_visitor_counts",
    ]
    for file_path in patterns_files:
        for chunk in pd.read_csv(
            file_path,
            dtype=str,
            chunksize=10000,
            usecols=lambda c: c in cols,
        ):
            if "placekey" not in chunk.columns:
                continue
            subset = chunk[chunk["placekey"].isin(needed_placekeys)]
            if subset.empty:
                continue
            for _, row in subset.iterrows():
                placekey = str(row.get("placekey") or "").strip()
                if not placekey:
                    continue
                candidate = {k: ("" if pd.isna(v) else str(v)) for k, v in row.items()}
                candidate["_source_file"] = str(file_path)
                if placekey not in lookup:
                    lookup[placekey] = candidate
                    continue
                if _score_pattern_row(candidate) > _score_pattern_row(lookup[placekey]):
                    lookup[placekey] = candidate
    return lookup


def _pick_display_name(rows: List[dict]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    example: Dict[str, str] = {}
    for row in rows:
        name = str(row.get("location_name") or "").strip()
        if not name:
            continue
        key = _norm(name)
        counts[key] += 1
        if key not in example or len(name) < len(example[key]):
            example[key] = name
    if not counts:
        return "unknown"
    top_key = sorted(counts.keys(), key=lambda k: (-counts[k], k))[0]
    return example[top_key]


def _build_popup_html(
    group_id: str,
    member_idx: int,
    member_row: dict,
    pattern_row: Optional[dict],
) -> str:
    def val(key: str, fallback_key: Optional[str] = None) -> str:
        if pattern_row and key in pattern_row and str(pattern_row.get(key) or "").strip():
            return str(pattern_row.get(key) or "").strip()
        if fallback_key:
            return str(member_row.get(fallback_key) or "").strip()
        return str(member_row.get(key) or "").strip()

    lat = val("latitude")
    lon = val("longitude")
    lines = [
        f"<b>{html.escape(group_id)} #{member_idx}</b>",
        f"placekey: {html.escape(str(member_row.get('placekey') or '').strip())}",
        f"safegraph_place_id: {html.escape(str(member_row.get('safegraph_place_id') or '').strip())}",
        f"name: {html.escape(val('location_name'))}",
        f"address: {html.escape(val('street_address'))}",
        f"city/state/zip: {html.escape(val('city'))} {html.escape(val('region'))} {html.escape(val('postal_code'))}",
        f"lat/lon: {html.escape(lat)} , {html.escape(lon)}",
        f"polygon_class: {html.escape(val('polygon_class'))}",
        f"wkt_area_sq_meters: {html.escape(val('wkt_area_sq_meters'))}",
        f"top_category: {html.escape(val('top_category'))}",
        f"sub_category: {html.escape(val('sub_category'))}",
    ]
    if pattern_row and pattern_row.get("_source_file"):
        lines.append(f"patterns_source: {html.escape(str(pattern_row.get('_source_file')))}")
    return "<br>".join(lines)


def _plot_group(
    group_id: str,
    members: List[dict],
    pattern_lookup: Dict[str, dict],
    out_file: Path,
) -> Tuple[int, int]:
    matched = 0
    polygon_count = 0

    lat_vals: List[float] = []
    lon_vals: List[float] = []
    for row in members:
        placekey = str(row.get("placekey") or "").strip()
        prow = pattern_lookup.get(placekey)
        lat = _safe_float(prow.get("latitude") if prow else row.get("latitude"))
        lon = _safe_float(prow.get("longitude") if prow else row.get("longitude"))
        if lat is not None and lon is not None:
            lat_vals.append(lat)
            lon_vals.append(lon)

    center_lat = sum(lat_vals) / len(lat_vals) if lat_vals else 36.0
    center_lon = sum(lon_vals) / len(lon_vals) if lon_vals else -95.0
    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=16, tiles="OpenStreetMap")

    bounds_points: List[List[float]] = []
    for idx, member in enumerate(members, start=1):
        color = PALETTE[(idx - 1) % len(PALETTE)]
        placekey = str(member.get("placekey") or "").strip()
        prow = pattern_lookup.get(placekey)
        if prow is not None:
            matched += 1

        name = str(
            (prow.get("location_name") if prow else "") or member.get("location_name") or "Unknown"
        ).strip()
        lat = _safe_float((prow.get("latitude") if prow else None) or member.get("latitude"))
        lon = _safe_float((prow.get("longitude") if prow else None) or member.get("longitude"))
        polygon_wkt = str((prow.get("polygon_wkt") if prow else "") or "").strip()
        popup_html = _build_popup_html(group_id, idx, member, prow)

        layer_name = f"#{idx} {name} [{placekey}]"
        feature = folium.FeatureGroup(name=layer_name, show=True)

        if lat is not None and lon is not None:
            folium.CircleMarker(
                location=[lat, lon],
                radius=5,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.95,
                popup=folium.Popup(popup_html, max_width=520),
                tooltip=layer_name,
            ).add_to(feature)
            bounds_points.append([lat, lon])

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
                    popup=folium.Popup(popup_html, max_width=520),
                    tooltip=layer_name,
                ).add_to(feature)
                minx, miny, maxx, maxy = geom.bounds
                bounds_points.extend([[miny, minx], [maxy, maxx]])
                polygon_count += 1
            except Exception:
                pass

        feature.add_to(fmap)

    if bounds_points:
        fmap.fit_bounds(bounds_points, padding=(20, 20))
    folium.LayerControl(collapsed=False).add_to(fmap)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_file))
    return matched, polygon_count


def _write_index_html(out_dir: Path, rows: List[dict]) -> Path:
    rows_html = []
    for row in rows:
        link = html.escape(str(row["file_name"]))
        gid = html.escape(str(row["group_id"]))
        name = html.escape(str(row["display_name"]))
        rows_html.append(
            "<tr>"
            f"<td><a href=\"{link}\" target=\"_blank\" rel=\"noopener\">{gid}</a></td>"
            f"<td>{name}</td>"
            f"<td>{row['members_total']}</td>"
            f"<td>{row['members_with_pattern']}</td>"
            f"<td>{row['members_with_polygon']}</td>"
            f"<td>{html.escape(str(row['regions']))}</td>"
            "</tr>"
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Duplicate Group OSM Maps</title>
  <style>
    body {{
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      margin: 24px;
      background: #f8fafc;
      color: #0f172a;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: white;
    }}
    th, td {{
      border: 1px solid #dbe2ea;
      padding: 8px;
      text-align: left;
      font-size: 13px;
    }}
    th {{
      background: #eef4fa;
    }}
  </style>
</head>
<body>
  <h1>Duplicate Group OSM Maps</h1>
  <p>Generated maps: {len(rows)}</p>
  <table>
    <thead>
      <tr>
        <th>Group</th><th>Name</th><th>Members</th><th>Pattern Rows</th><th>Polygons</th><th>Regions</th>
      </tr>
    </thead>
    <tbody>
      {"".join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    out = out_dir / "index.html"
    out.write_text(html_doc, encoding="utf-8")
    return out


def main() -> int:
    args = parse_args()
    cluster_members_path = _resolve_user_path(args.cluster_members)
    output_dir = _resolve_user_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cluster_members_path.exists():
        raise FileNotFoundError(f"cluster_members.csv not found: {cluster_members_path}")

    patterns_files = _resolve_patterns_files(args)
    if not patterns_files:
        raise FileNotFoundError(
            "No patterns files found. Provide --patterns-file or --patterns-glob."
        )

    groups_all = _read_cluster_members(cluster_members_path)
    groups = _filter_groups(
        groups=groups_all,
        min_members=max(2, int(args.min_members)),
        group_ids=args.group_id,
        name_filter=args.name_filter,
        regions=args.region,
    )
    if not groups:
        print("No matching groups after filters.")
        return 0

    needed_placekeys = {
        str(row.get("placekey") or "").strip()
        for rows in groups.values()
        for row in rows
        if str(row.get("placekey") or "").strip()
    }
    pattern_lookup = _load_pattern_lookup(patterns_files, needed_placekeys)

    ordered_groups = sorted(groups.items(), key=lambda kv: kv[0])
    if args.max_groups and args.max_groups > 0:
        ordered_groups = ordered_groups[: args.max_groups]

    index_rows: List[dict] = []
    generated = 0
    skipped = 0

    for group_id, members in ordered_groups:
        display_name = _pick_display_name(members)
        members_with_pattern = sum(
            1
            for m in members
            if str(m.get("placekey") or "").strip() in pattern_lookup
        )
        if members_with_pattern < max(1, int(args.min_members_with_pattern)):
            skipped += 1
            continue

        file_name = f"{group_id}-{_slug(display_name)}.html"
        out_file = output_dir / file_name
        matched, polygon_count = _plot_group(
            group_id=group_id,
            members=members,
            pattern_lookup=pattern_lookup,
            out_file=out_file,
        )

        regions = sorted({str(m.get("region") or "").strip() for m in members if str(m.get("region") or "").strip()})
        index_rows.append(
            {
                "group_id": group_id,
                "display_name": display_name,
                "file_name": file_name,
                "members_total": len(members),
                "members_with_pattern": matched,
                "members_with_polygon": polygon_count,
                "regions": ",".join(regions),
            }
        )
        generated += 1

    index_path = _write_index_html(output_dir, index_rows)
    print(f"Cluster file: {cluster_members_path}")
    print(f"Patterns files used: {len(patterns_files)}")
    print(f"Groups considered: {len(ordered_groups)}")
    print(f"Maps generated: {generated}")
    print(f"Groups skipped (insufficient pattern matches): {skipped}")
    print(f"Output dir: {output_dir}")
    print(f"Index: {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
