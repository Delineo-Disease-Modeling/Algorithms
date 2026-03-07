#!/usr/bin/env python3
"""
Build an interactive CZ map where clicking a CBG polygon updates a side panel
with that CBG's population and POI list.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import folium
import geopandas as gpd
import pandas as pd


STATE_FIPS_TO_ABBR = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}


def normalize_cbg(raw: object) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        text = str(int(float(text)))
    except (TypeError, ValueError):
        pass
    if len(text) == 11 and text.isdigit():
        text = text.zfill(12)
    if len(text) == 12 and text.isdigit():
        return text
    return None


def normalize_int(raw: object, default: int = 0) -> int:
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def normalize_str(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, float) and pd.isna(raw):
        return ""
    text = str(raw).strip()
    return "" if text.lower() == "nan" else text


def load_cbg_population_map(cbg_csv: Path) -> Dict[str, int]:
    df = pd.read_csv(cbg_csv, dtype=str)
    if "cbg" not in df.columns:
        raise ValueError(f"Missing required column 'cbg' in {cbg_csv}")
    pop_col = "population" if "population" in df.columns else None
    if pop_col is None:
        raise ValueError(f"Missing required column 'population' in {cbg_csv}")

    mapping: Dict[str, int] = {}
    for _, row in df.iterrows():
        cbg = normalize_cbg(row.get("cbg"))
        if not cbg:
            continue
        mapping[cbg] = normalize_int(row.get(pop_col), default=0)
    if not mapping:
        raise ValueError(f"No valid CBG rows found in {cbg_csv}")
    return mapping


def load_state_geometries(server_dir: Path, state_fips: str) -> gpd.GeoDataFrame:
    shp_2016 = (
        server_dir
        / "data"
        / "shapefiles_2016"
        / f"tl_2016_{state_fips}_bg"
        / f"tl_2016_{state_fips}_bg.shp"
    )
    if not shp_2016.exists():
        state_abbr = STATE_FIPS_TO_ABBR.get(state_fips)
        print(
            "WARNING: Missing 2016 TIGER/Line shapefile for "
            f"state FIPS {state_fips} (abbr: {state_abbr}) at {shp_2016}. "
            "Legacy fallback is disabled."
        )
        raise FileNotFoundError(
            f"Could not find required 2016 shapefile for state FIPS {state_fips}: {shp_2016}"
        )

    gdf = gpd.read_file(shp_2016)
    source = str(shp_2016)

    if "CensusBlockGroup" not in gdf.columns and "GEOID" in gdf.columns:
        gdf["CensusBlockGroup"] = gdf["GEOID"].astype(str)
    if "CensusBlockGroup" not in gdf.columns:
        raise ValueError(f"Missing CBG key columns in geometry source {source}")

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    else:
        gdf = gdf.to_crs("EPSG:4326")

    gdf["CensusBlockGroup"] = gdf["CensusBlockGroup"].astype(str).str.zfill(12)
    return gdf[["CensusBlockGroup", "geometry"]].copy()


def load_cluster_geometries(server_dir: Path, cbgs: List[str]) -> gpd.GeoDataFrame:
    state_fips_codes = sorted({cbg[:2] for cbg in cbgs})
    parts = [load_state_geometries(server_dir, sfips) for sfips in state_fips_codes]
    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
    return merged[merged["CensusBlockGroup"].isin(set(cbgs))].copy()


def load_poi_map(poi_csv: Path, cluster_cbgs: List[str]) -> Dict[str, List[Dict[str, object]]]:
    wanted_cols = [
        "poi_cbg",
        "placekey",
        "location_name",
        "top_category",
        "sub_category",
        "street_address",
        "city",
        "region",
        "postal_code",
        "raw_visit_counts",
        "raw_visitor_counts",
    ]
    df = pd.read_csv(poi_csv, dtype=str, usecols=lambda c: c in wanted_cols)
    if "poi_cbg" not in df.columns:
        raise ValueError(f"Missing required column 'poi_cbg' in {poi_csv}")

    cluster_set = set(cluster_cbgs)
    poi_map: Dict[str, List[Dict[str, object]]] = {cbg: [] for cbg in cluster_cbgs}

    for _, row in df.iterrows():
        cbg = normalize_cbg(row.get("poi_cbg"))
        if not cbg or cbg not in cluster_set:
            continue
        poi_map[cbg].append(
            {
                "placekey": normalize_str(row.get("placekey")),
                "name": normalize_str(row.get("location_name")) or "Unknown POI",
                "top_category": normalize_str(row.get("top_category")),
                "sub_category": normalize_str(row.get("sub_category")),
                "street_address": normalize_str(row.get("street_address")),
                "city": normalize_str(row.get("city")),
                "region": normalize_str(row.get("region")),
                "postal_code": normalize_str(row.get("postal_code")),
                "raw_visit_counts": normalize_int(row.get("raw_visit_counts"), default=0),
                "raw_visitor_counts": normalize_int(row.get("raw_visitor_counts"), default=0),
            }
        )

    for cbg in poi_map:
        poi_map[cbg].sort(
            key=lambda x: (x["raw_visit_counts"], x["raw_visitor_counts"], x["name"]),
            reverse=True,
        )
    return poi_map


def build_feature_collection(cluster_gdf: gpd.GeoDataFrame, pop_map: Dict[str, int], seed_cbg: Optional[str]):
    features = []
    for _, row in cluster_gdf.iterrows():
        cbg = row["CensusBlockGroup"]
        features.append(
            {
                "type": "Feature",
                "geometry": row.geometry.__geo_interface__,
                "properties": {
                    "cbg": cbg,
                    "population": int(pop_map.get(cbg, 0)),
                    "is_seed": bool(seed_cbg and cbg == seed_cbg),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def inject_sidebar_ui(
    m: folium.Map,
    geojson_layer_name: str,
    cbg_meta: Dict[str, Dict[str, object]],
    cbg_pois: Dict[str, List[Dict[str, object]]],
    initial_cbg: Optional[str],
    title: str,
    include_pois_tab: bool = True,
) -> None:
    poi_tab_button = (
        '<button class="tab-btn" data-tab="pois">POIs</button>'
        if include_pois_tab
        else ""
    )
    poi_tab_panel = (
        '<div id="tab-pois" class="tab-content"><div id="pois-body" class="muted">Click a CBG polygon to view POIs.</div></div>'
        if include_pois_tab
        else ""
    )
    panel_html = f"""
    <style>
      #cbg-sidebar {{
        position: absolute;
        top: 12px;
        right: 12px;
        width: 380px;
        max-height: calc(100% - 24px);
        z-index: 9999;
        background: #ffffff;
        border: 1px solid #d0d7de;
        border-radius: 10px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        overflow: hidden;
        display: flex;
        flex-direction: column;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }}
      #cbg-sidebar .header {{
        padding: 10px 12px;
        border-bottom: 1px solid #eceff3;
        font-weight: 700;
        font-size: 14px;
      }}
      #cbg-sidebar .tabs {{
        display: flex;
        border-bottom: 1px solid #eceff3;
      }}
      #cbg-sidebar .tab-btn {{
        flex: 1;
        padding: 8px 10px;
        border: 0;
        background: #f6f8fa;
        cursor: pointer;
        font-size: 13px;
      }}
      #cbg-sidebar .tab-btn.active {{
        background: #ffffff;
        border-bottom: 2px solid #2563eb;
        font-weight: 600;
      }}
      #cbg-sidebar .tab-content {{
        display: none;
        padding: 10px 12px;
        overflow: auto;
        font-size: 13px;
      }}
      #cbg-sidebar .tab-content.active {{
        display: block;
        height: 100%;
      }}
      #cbg-sidebar .muted {{
        color: #6b7280;
      }}
      #cbg-sidebar table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
      }}
      #cbg-sidebar th, #cbg-sidebar td {{
        border-bottom: 1px solid #eef2f7;
        padding: 6px 4px;
        text-align: left;
        vertical-align: top;
      }}
      #cbg-sidebar th {{
        background: #f8fafc;
        position: sticky;
        top: 0;
        z-index: 1;
      }}
      #cbg-sidebar .pill {{
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        border-radius: 9999px;
        padding: 2px 8px;
        font-size: 12px;
        margin-right: 4px;
      }}
      .map-title-box {{
        position: absolute;
        top: 12px;
        left: 12px;
        z-index: 9998;
        background: rgba(255, 255, 255, 0.93);
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 8px 10px;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 13px;
        font-weight: 600;
      }}
    </style>
    <div id="cbg-sidebar">
      <div class="header">CBG Inspector</div>
      <div class="tabs">
        <button class="tab-btn active" data-tab="details">Details</button>
        {poi_tab_button}
      </div>
      <div id="tab-details" class="tab-content active">
        <div id="details-body" class="muted">Click a CBG polygon to view details.</div>
      </div>
      {poi_tab_panel}
    </div>
    <div class="map-title-box">{title}</div>
    """
    m.get_root().html.add_child(folium.Element(panel_html))

    script = f"""
    <script>
      const cbgMeta = {json.dumps(cbg_meta)};
      const cbgPois = {json.dumps(cbg_pois)};
      const initialCbg = {json.dumps(initial_cbg)};
      const includePoisTab = {json.dumps(include_pois_tab)};
      const geoLayer = {geojson_layer_name};

      function esc(v) {{
        return String(v ?? '')
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#39;');
      }}

      function fmtInt(v) {{
        const n = Number(v);
        if (!Number.isFinite(n)) return '0';
        return n.toLocaleString();
      }}

      function renderDetails(cbg) {{
        const body = document.getElementById('details-body');
        const meta = cbgMeta[cbg];
        if (!meta) {{
          body.innerHTML = '<span class="muted">No metadata found for CBG ' + esc(cbg) + '.</span>';
          return;
        }}
        const seedPill = meta.is_seed ? '<span class="pill">Seed</span>' : '';
        body.innerHTML = `
          <div style="margin-bottom:8px;">${{seedPill}}<strong>CBG:</strong> ${{esc(cbg)}}</div>
          <div><strong>Population:</strong> ${{fmtInt(meta.population)}}</div>
          <div style="margin-top:8px;" class="muted">
            ${{
              includePoisTab
                ? 'Click POIs tab for place-level details.'
                : 'Population shown for the selected CBG.'
            }}
          </div>
        `;
      }}

      function renderPois(cbg) {{
        const body = document.getElementById('pois-body');
        const pois = cbgPois[cbg] || [];
        if (pois.length === 0) {{
          body.innerHTML = '<span class="muted">No POIs for CBG ' + esc(cbg) + '.</span>';
          return;
        }}
        let rows = '';
        for (const poi of pois) {{
          const category = poi.sub_category || poi.top_category || '';
          const addr = [poi.street_address, poi.city, poi.region, poi.postal_code].filter(Boolean).join(', ');
          rows += `
            <tr>
              <td>
                <div><strong>${{esc(poi.name)}}</strong></div>
                <div class="muted">${{esc(category)}}</div>
                <div class="muted">${{esc(addr)}}</div>
              </td>
              <td>${{fmtInt(poi.raw_visit_counts)}}</td>
              <td>${{fmtInt(poi.raw_visitor_counts)}}</td>
            </tr>
          `;
        }}
        body.innerHTML = `
          <div style="margin-bottom:8px;"><strong>CBG:</strong> ${{esc(cbg)}} | <strong>POIs:</strong> ${{fmtInt(pois.length)}}</div>
          <table>
            <thead>
              <tr>
                <th>POI</th>
                <th>Visits</th>
                <th>Visitors</th>
              </tr>
            </thead>
            <tbody>${{rows}}</tbody>
          </table>
        `;
      }}

      function activateTab(tabName) {{
        document.querySelectorAll('#cbg-sidebar .tab-btn').forEach(btn => {{
          btn.classList.toggle('active', btn.dataset.tab === tabName);
        }});
        document.querySelectorAll('#cbg-sidebar .tab-content').forEach(panel => {{
          panel.classList.toggle('active', panel.id === 'tab-' + tabName);
        }});
      }}

      document.querySelectorAll('#cbg-sidebar .tab-btn').forEach(btn => {{
        btn.addEventListener('click', () => activateTab(btn.dataset.tab));
      }});

      let selectedLayer = null;
      function styleSelected(layer) {{
        layer.setStyle({{
          color: '#111827',
          weight: 3,
          fillOpacity: 0.7
        }});
      }}
      function clearSelected() {{
        if (!selectedLayer) return;
        geoLayer.resetStyle(selectedLayer);
        selectedLayer = null;
      }}
      function selectCbg(cbg) {{
        renderDetails(cbg);
        if (includePoisTab) {{
          renderPois(cbg);
        }}
      }}

      geoLayer.eachLayer(layer => {{
        layer.on('click', () => {{
          const cbg = layer.feature.properties.cbg;
          clearSelected();
          selectedLayer = layer;
          styleSelected(layer);
          selectCbg(cbg);
        }});
      }});

      if (initialCbg) {{
        let initialLayer = null;
        geoLayer.eachLayer(layer => {{
          if (layer.feature && layer.feature.properties && layer.feature.properties.cbg === initialCbg) {{
            initialLayer = layer;
          }}
        }});
        if (initialLayer) {{
          selectedLayer = initialLayer;
          styleSelected(initialLayer);
          selectCbg(initialCbg);
        }}
      }}
    </script>
    """
    m.get_root().html.add_child(folium.Element(script))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a CZ map with click-to-inspect CBG population and POIs in a side tab panel."
    )
    parser.add_argument("--cbg-csv", required=True, help="CSV with columns: cbg,population")
    parser.add_argument("--poi-csv", required=True, help="POI CSV with poi_cbg and metadata columns")
    parser.add_argument("--output-html", required=True, help="Output HTML path")
    parser.add_argument("--seed-cbg", default=None, help="Optional seed CBG for initial selection")
    parser.add_argument(
        "--details-only",
        action="store_true",
        help="Show only CBG details (population) and hide POI tab.",
    )
    parser.add_argument(
        "--title",
        default="Convenience Zone Map (Click a CBG)",
        help="Title shown in map overlay",
    )
    args = parser.parse_args()

    cbg_csv = Path(args.cbg_csv).resolve()
    poi_csv = Path(args.poi_csv).resolve()
    output_html = Path(args.output_html).resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)

    server_dir = Path(__file__).resolve().parents[1]
    pop_map = load_cbg_population_map(cbg_csv)
    cbgs = sorted(pop_map.keys())
    cluster_gdf = load_cluster_geometries(server_dir, cbgs)
    if cluster_gdf.empty:
        raise RuntimeError("No matching CBG geometries found for the provided CBG list.")

    seed_cbg = normalize_cbg(args.seed_cbg) if args.seed_cbg else None
    include_pois_tab = not args.details_only
    cbg_pois = load_poi_map(poi_csv, cbgs) if include_pois_tab else {cbg: [] for cbg in cbgs}
    cbg_meta = {
        cbg: {"population": int(pop_map.get(cbg, 0)), "is_seed": bool(seed_cbg and cbg == seed_cbg)}
        for cbg in cbgs
    }

    feature_collection = build_feature_collection(cluster_gdf, pop_map, seed_cbg)

    if seed_cbg and seed_cbg in set(cluster_gdf["CensusBlockGroup"]):
        center_pt = (
            cluster_gdf[cluster_gdf["CensusBlockGroup"] == seed_cbg]
            .representative_point()
            .iloc[0]
        )
    else:
        center_pt = cluster_gdf.representative_point().iloc[0]

    m = folium.Map(location=[center_pt.y, center_pt.x], zoom_start=11, tiles="OpenStreetMap")
    geojson = folium.GeoJson(
        feature_collection,
        name="cz_cbgs",
        style_function=lambda f: {
            "fillColor": "#16a34a" if f["properties"].get("is_seed") else "#2563eb",
            "color": "#0f172a",
            "weight": 1.2,
            "fillOpacity": 0.45,
        },
        highlight_function=lambda _f: {"weight": 2.2, "color": "#111827", "fillOpacity": 0.62},
        tooltip=folium.GeoJsonTooltip(
            fields=["cbg", "population"],
            aliases=["CBG", "Population"],
            localize=True,
            sticky=False,
        ),
    )
    geojson.add_to(m)

    minx, miny, maxx, maxy = cluster_gdf.total_bounds
    m.fit_bounds([[miny, minx], [maxy, maxx]])

    inject_sidebar_ui(
        m=m,
        geojson_layer_name=geojson.get_name(),
        cbg_meta=cbg_meta,
        cbg_pois=cbg_pois,
        initial_cbg=seed_cbg or cbgs[0],
        title=args.title,
        include_pois_tab=include_pois_tab,
    )

    m.save(str(output_html))
    print(output_html)


if __name__ == "__main__":
    main()
