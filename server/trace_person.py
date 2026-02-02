#!/usr/bin/env python3
"""
Trace one or more people through a patterns.json run and output CSV + HTML timeline,
with an optional Folium map showing markers and dotted path.

Usage example:
  venv/bin/python server/trace_person.py \
    --papdata output/barnsdall_2025_5000/papdata.json \
    --patterns output/barnsdall_2025_5000/patterns.json \
    --person-id 12 --person-id 87 \
    --start-time 2025-07-01T00:00:00 \
    --out-dir output/barnsdall_2025_5000 \
    --map --map-only-changes --heatmap
"""

import argparse
import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple


def _parse_person_ids(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        if not v:
            continue
        parts = [p.strip() for p in v.split(',') if p.strip()]
        out.extend(parts)
    # Deduplicate preserving order
    seen = set()
    uniq = []
    for p in out:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _location_label(papdata: Dict[str, Any], loc_type: str, loc_id: str) -> str:
    if loc_type == "home":
        home = papdata.get("homes", {}).get(str(loc_id), {})
        cbg = home.get("cbg")
        return f"Home {loc_id}" + (f" (CBG {cbg})" if cbg else "")
    place = papdata.get("places", {}).get(str(loc_id), {})
    label = place.get("label") or place.get("location_name") or "Unknown place"
    return f"{label} (place {loc_id})"


def _find_locations_for_timestep(timestep: Dict[str, Any], target_set: set) -> Dict[str, Tuple[str, str]]:
    """
    Returns mapping: person_id -> (loc_type, loc_id)
    loc_type: "home" or "place"
    """
    found: Dict[str, Tuple[str, str]] = {}
    homes = timestep.get("homes", {}) or {}
    for home_id, people in homes.items():
        for pid in people:
            if pid in target_set:
                found[str(pid)] = ("home", str(home_id))
    places = timestep.get("places", {}) or {}
    for place_id, people in places.items():
        for pid in people:
            if pid in target_set:
                found[str(pid)] = ("place", str(place_id))
    return found


def _collapse_segments(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse consecutive rows with same location for a person."""
    if not rows:
        return rows
    out = [rows[0].copy()]
    for r in rows[1:]:
        last = out[-1]
        if r["person_id"] == last["person_id"] and r["loc_type"] == last["loc_type"] and r["loc_id"] == last["loc_id"]:
            last["end_minute"] = r["minute"]
            last["end_time"] = r["time"]
        else:
            out.append(r.copy())
    for seg in out:
        if "end_minute" not in seg:
            seg["end_minute"] = seg["minute"]
            seg["end_time"] = seg["time"]
    return out


def _load_cbg_centroids(path: str) -> Dict[str, Tuple[float, float]]:
    if not path or not os.path.exists(path):
        return {}
    try:
        import yaml
    except Exception:
        return {}
    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except Exception:
        return {}

    centroids: Dict[str, Tuple[float, float]] = {}

    def _extract(entry: Dict[str, Any]):
        cbg = entry.get("GEOID10") or entry.get("cbg") or entry.get("cbg_id")
        lat = entry.get("lat") or entry.get("latitude") or entry.get("centroid_lat")
        lon = entry.get("lon") or entry.get("lng") or entry.get("longitude") or entry.get("centroid_lon")
        if cbg is None or lat is None or lon is None:
            return
        try:
            centroids[str(cbg)] = (float(lat), float(lon))
        except Exception:
            return

    if isinstance(raw, dict):
        for cbg, entry in raw.items():
            if isinstance(entry, dict):
                lat = entry.get("lat") or entry.get("latitude") or entry.get("centroid_lat")
                lon = entry.get("lon") or entry.get("lng") or entry.get("longitude") or entry.get("centroid_lon")
                if lat is None or lon is None:
                    continue
                try:
                    centroids[str(cbg)] = (float(lat), float(lon))
                except Exception:
                    continue
    elif isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                _extract(entry)

    return centroids


def _add_map_legend(fmap, person_colors: Dict[str, str], show_heatmap_note: bool):
    items = "".join(
        f"<div style='margin:2px 0;'><span style='display:inline-block;width:12px;height:12px;background:{color};margin-right:6px;'></span>{pid}</div>"
        for pid, color in person_colors.items()
    )
    heat_note = "<div style='margin-top:6px;font-size:12px;'>Heatmap: density of location points (each point weight=1)</div>" if show_heatmap_note else ""
    html = f"""
    <div style="
        position: fixed;
        bottom: 20px;
        left: 20px;
        z-index: 9999;
        background: white;
        border: 1px solid #ccc;
        padding: 8px 10px;
        border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        font-family: Arial, sans-serif;
        font-size: 13px;">
      <div style="font-weight:bold;margin-bottom:6px;">Map Key</div>
      <div>Person colors:</div>
      {items}
      {heat_note}
    </div>
    """
    try:
        from folium import Element
        fmap.get_root().html.add_child(Element(html))
    except Exception:
        pass


def _get_location_point(papdata: Dict[str, Any], centroids: Dict[str, Tuple[float, float]], loc_type: str, loc_id: str):
    if loc_type == "place":
        place = papdata.get("places", {}).get(str(loc_id), {})
        try:
            lat = float(place.get("latitude"))
            lon = float(place.get("longitude"))
            return (lat, lon)
        except Exception:
            return None
    if loc_type == "home":
        home = papdata.get("homes", {}).get(str(loc_id), {})
        cbg = home.get("cbg")
        if cbg is None:
            return None
        return centroids.get(str(cbg))
    return None


def main():
    parser = argparse.ArgumentParser(description="Trace person movement through patterns.json")
    parser.add_argument("--papdata", required=True, help="Path to papdata.json")
    parser.add_argument("--patterns", required=True, help="Path to patterns.json")
    parser.add_argument("--person-id", action="append", default=[], help="Person id to trace (repeat or comma-separated)")
    parser.add_argument("--start-time", default=None, help="ISO start time (e.g. 2025-07-01T00:00:00)")
    parser.add_argument("--out-dir", default="output", help="Directory for outputs")
    parser.add_argument("--only-changes", action="store_true", help="Collapse consecutive identical locations")
    parser.add_argument("--map", action="store_true", help="Generate map HTML")
    parser.add_argument("--map-out", default=None, help="Map output path (default trace_people_map.html in out-dir)")
    parser.add_argument("--map-only-changes", action="store_true", help="Use collapsed rows for map")
    parser.add_argument("--map-sample-step", type=int, default=1, help="Keep every Nth row for map (default 1)")
    parser.add_argument("--max-markers", type=int, default=2000, help="Max markers to plot (default 2000)")
    parser.add_argument("--heatmap", action="store_true", help="Add heatmap layer")
    parser.add_argument("--cbg-centroids", default=None, help="Optional YAML with CBG centroids for home locations")
    args = parser.parse_args()

    person_ids = _parse_person_ids(args.person_id)
    if not person_ids:
        raise SystemExit("No person ids provided. Use --person-id.")

    with open(args.papdata, "r") as f:
        papdata = json.load(f)
    with open(args.patterns, "r") as f:
        patterns = json.load(f)

    start_time = None
    if args.start_time:
        start_time = datetime.fromisoformat(args.start_time)

    # Sort timesteps by minute
    minutes = sorted([int(k) for k in patterns.keys()])
    target_set = set(person_ids)

    rows: List[Dict[str, Any]] = []
    for minute in minutes:
        step = patterns.get(str(minute), {})
        found = _find_locations_for_timestep(step, target_set)
        ts = (start_time + timedelta(minutes=minute)).isoformat() if start_time else str(minute)
        for pid in person_ids:
            loc = found.get(pid)
            if loc is None:
                loc_type, loc_id = "unknown", ""
                label = "Unknown"
            else:
                loc_type, loc_id = loc
                label = _location_label(papdata, loc_type, loc_id)
            rows.append({
                "person_id": pid,
                "minute": minute,
                "time": ts,
                "loc_type": loc_type,
                "loc_id": loc_id,
                "label": label,
            })

    if args.only_changes:
        collapsed: List[Dict[str, Any]] = []
        for pid in person_ids:
            person_rows = [r for r in rows if r["person_id"] == pid]
            collapsed.extend(_collapse_segments(person_rows))
        rows = collapsed

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "trace_people.csv")
    html_path = os.path.join(args.out_dir, "trace_people.html")

    # Write CSV
    with open(csv_path, "w") as f:
        f.write("person_id,minute,time,loc_type,loc_id,label\n")
        for r in rows:
            label = r["label"].replace('"', '""')
            f.write(f"{r['person_id']},{r['minute']},{r['time']},{r['loc_type']},{r['loc_id']},\"{label}\"\n")

    # Write simple HTML table
    def row_to_html(r):
        return (
            f"<tr>"
            f"<td>{r['person_id']}</td>"
            f"<td>{r['minute']}</td>"
            f"<td>{r['time']}</td>"
            f"<td>{r['loc_type']}</td>"
            f"<td>{r['loc_id']}</td>"
            f"<td>{r['label']}</td>"
            f"</tr>"
        )

    with open(html_path, "w") as f:
        f.write("<html><head><meta charset='utf-8'>"
                "<title>Person Trace</title>"
                "<style>body{font-family:Arial, sans-serif;}"
                "table{border-collapse:collapse;width:100%;}"
                "th,td{border:1px solid #ddd;padding:6px;}"
                "th{background:#f2f2f2;}</style>"
                "</head><body>")
        f.write(f"<h2>Trace for person(s): {', '.join(person_ids)}</h2>")
        f.write("<table><thead><tr>"
                "<th>Person</th><th>Minute</th><th>Time</th><th>Type</th><th>ID</th><th>Label</th>"
                "</tr></thead><tbody>")
        for r in rows:
            f.write(row_to_html(r))
        f.write("</tbody></table></body></html>")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {html_path}")

    if args.map:
        try:
            import folium
            from folium import plugins
        except Exception:
            raise SystemExit("folium is required for --map. Install it in your venv.")

        centroids = _load_cbg_centroids(args.cbg_centroids) if args.cbg_centroids else {}

        map_rows = rows
        if args.map_only_changes:
            collapsed: List[Dict[str, Any]] = []
            for pid in person_ids:
                person_rows = [r for r in rows if r["person_id"] == pid]
                collapsed.extend(_collapse_segments(person_rows))
            map_rows = collapsed

        if args.map_sample_step > 1:
            map_rows = [r for idx, r in enumerate(map_rows) if idx % args.map_sample_step == 0]

        # Build per-person points
        points_by_person: Dict[str, List[Tuple[float, float, str, str]]] = {pid: [] for pid in person_ids}
        for r in map_rows:
            loc_type = r["loc_type"]
            if loc_type == "unknown":
                continue
            pt = _get_location_point(papdata, centroids, loc_type, r["loc_id"])
            if not pt:
                continue
            points_by_person[r["person_id"]].append((pt[0], pt[1], r["time"], r["label"]))

        # Flatten for centering
        all_points = [p for plist in points_by_person.values() for p in plist]
        if not all_points:
            raise SystemExit("No mappable points found (missing lat/lon or centroids).")
        avg_lat = sum(p[0] for p in all_points) / len(all_points)
        avg_lon = sum(p[1] for p in all_points) / len(all_points)

        fmap = folium.Map(location=[avg_lat, avg_lon], zoom_start=12)

        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf"
        ]

        total_markers = 0
        cluster = plugins.MarkerCluster().add_to(fmap)
        person_colors = {}
        for idx, pid in enumerate(person_ids):
            color = colors[idx % len(colors)]
            person_colors[pid] = color
            pts = points_by_person.get(pid, [])
            if not pts:
                continue

            # Dotted path
            path_coords = [(p[0], p[1]) for p in pts]
            folium.PolyLine(path_coords, color=color, weight=3, opacity=0.8, dash_array="5,8").add_to(fmap)

            for lat, lon, ts, label in pts:
                if total_markers >= args.max_markers:
                    break
                popup = folium.Popup(f"Person {pid}<br>{ts}<br>{label}", max_width=300)
                folium.CircleMarker(location=[lat, lon], radius=4, color=color, fill=True, fill_opacity=0.9, popup=popup).add_to(cluster)
                total_markers += 1

        if args.heatmap:
            heat_points = [[p[0], p[1], 1] for p in all_points]
            plugins.HeatMap(heat_points, radius=12, blur=18).add_to(fmap)

        _add_map_legend(fmap, person_colors, args.heatmap)

        map_out = args.map_out or os.path.join(args.out_dir, "trace_people_map.html")
        fmap.save(map_out)
        print(f"Wrote: {map_out}")


if __name__ == "__main__":
    main()
