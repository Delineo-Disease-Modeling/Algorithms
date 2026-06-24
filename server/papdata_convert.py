"""Convert a synthetic-population DataFrame into the papdata dict the simulator
reads (people / homes / places). Splits the per-POI place bundle out of the
SafeGraph patterns data, including the area fill and the catchment fraction f_j.

Extracted from `popgen.py` (which re-exports `convert_data` and
`CATCHMENT_FJ_FLOOR`). The shared catchment helpers come from `patterns` so the
places-bundle f_j and the movement-target f_j use one definition.
"""
import pandas as pd

try:
    from shapely import wkt as shapely_wkt
except ImportError:
    shapely_wkt = None

# Catchment helpers shared with gen_patterns so the places-bundle f_j matches the
# movement-target f_j (same definition + same per-run median fallback).
from patterns import _catchment_fraction, _median_fj_fallback

# Lower bound on the emitted per-POI catchment fraction f_j. The simulator's
# external-FOI term scales as (1 - f_j)/f_j, which diverges as f_j -> 0, so a data
# artifact (e.g. one in-cluster visitor out of thousands) would otherwise produce
# absurd external pressure. 0.02 caps the ratio at 49x. See
# docs/MOVEMENT_MODEL_REDESIGN.md §10.
CATCHMENT_FJ_FLOOR = 0.02


def convert_data(df, cz_data, shared_data=None):
    """
    Convert data frame with person and household information into a specific dictionary format.

    Args:
        df: DataFrame with person/household data
        cz_data: Dictionary mapping CBG IDs to population counts
        shared_data: Pre-loaded PatternsData used to derive the places dict.
            If None/empty, the output will contain zero places.
    """
    # Initialize output dictionary
    output = {
        "people": {},
        "homes": {},
        "places": {}
    }

    # Create mappings
    # For sex: M -> 0, F -> 1
    sex_mapping = {"M": 0, "F": 1}

    # Process each row in the dataframe
    for index, row in df.iterrows():
        person_id = str(row['person_id'])
        household_id = str(row['household_id'])

        # Add person to people dictionary
        output["people"][person_id] = {
            "sex": sex_mapping.get(row['gender']),
            "age": row['age'],
            "home": household_id
        }

        # Add or update household in homes dictionary
        if household_id not in output["homes"]:
            home_data = {
                "cbg": row['cbg'],
                "members": 1
            }
            # Add coordinates if available
            if pd.notna(row.get('household_lat')) and pd.notna(row.get('household_lon')):
                home_data["latitude"] = row['household_lat']
                home_data["longitude"] = row['household_lon']
            output["homes"][household_id] = home_data
        else:
            output["homes"][household_id]["members"] += 1

    # Get places from the patterns data
    cbgs = list(cz_data.keys())
    cbg_set = set(cbgs)
    metadata_cols = [
        'location_name',
        'top_category',
        'latitude',
        'longitude',
        'street_address',
        'postal_code',
        'polygon_wkt',
        'wkt_area_sq_meters',
        'visitor_home_cbgs',
    ]

    if shared_data is not None and not shared_data.is_empty():
        placekeys = shared_data.get_placekeys_for_cbgs(cbg_set)
        places = shared_data.for_popgen_places(placekeys)
    else:
        places = pd.DataFrame(columns=['placekey'] + metadata_cols)

    for col in metadata_cols:
        if col not in places.columns:
            places[col] = None

    def _coerce_coord(v):
        """Force lat/lon into a float or None.

        pd.read_csv infers the column dtype from the file contents: a single
        non-numeric cell (empty string, "NA", stray text) promotes the whole
        column to object and every cell comes back as a str. The frontend
        map uses Number.isFinite which is strict and rejects strings, so we
        normalize to float here.
        """
        if v is None:
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f != f:  # NaN
            return None
        return f

    def _coerce_footprint(raw):
        if shapely_wkt is None or raw is None:
            return None
        text = str(raw).strip()
        if not text:
            return None
        try:
            geom = shapely_wkt.loads(text)
        except Exception:
            return None
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type not in ('Polygon', 'MultiPolygon'):
            return None
        try:
            return geom.__geo_interface__
        except Exception:
            return None

    def _clean_optional_text(v):
        if pd.isna(v):
            return None
        text = str(v).strip()
        return text or None

    # Physical floor area (m^2) per POI, from SafeGraph WKT_AREA_SQ_METERS. Used
    # downstream for area-aware Wells-Riley ventilation. Non-positive / missing
    # values are filled with the per-category median, then the global median,
    # then a constant (the dataset median POI is ~649 m^2). Winsorizing the long
    # tail (e.g. a region polygon logged as one giant "POI") happens at the
    # physics step in the simulator, not here.
    if 'wkt_area_sq_meters' in places.columns:
        _area_series = pd.to_numeric(places['wkt_area_sq_meters'], errors='coerce')
        _area_series = _area_series.where(_area_series > 0)
    else:
        _area_series = pd.Series([float('nan')] * len(places), index=places.index)
    _global_med_area = _area_series.median()
    if pd.isna(_global_med_area):
        _global_med_area = 650.0
    _cat_med_area = _area_series.groupby(places['top_category']).median().dropna().to_dict()

    # Catchment fraction f_j per POI: the share of the POI's observed visitors who
    # live inside our simulated cluster. Same definition + median fallback as
    # gen_patterns, so the movement targets and the external-FOI term agree; the
    # ~23% of low-traffic POIs without visitor_home_cbgs get the per-run median.
    # Floored (CATCHMENT_FJ_FLOOR) because the simulator's external term scales as
    # (1 - f_j)/f_j. (Behavior-inert until the simulator reads catchment_fj.)
    _fj_fallback = _median_fj_fallback(
        _catchment_fraction(v, cbg_set) for v in places.get('visitor_home_cbgs', pd.Series(dtype=object))
    )

    for i, row in places.iterrows():
        _area = _area_series.loc[i]
        if pd.isna(_area):
            _area = _cat_med_area.get(row['top_category'], _global_med_area)
        _fj = _catchment_fraction(row.get('visitor_home_cbgs'), cbg_set)
        if _fj is None:
            _fj = _fj_fallback
        output['places'][str(i)] = {
            'placekey': row['placekey'],
            'label': row['location_name'],
            'latitude': _coerce_coord(row['latitude']),
            'longitude': _coerce_coord(row['longitude']),
            # Sometimes this is empty
            'top_category': 'None' if pd.isna(row['top_category']) else row['top_category'],
            'street_address': _clean_optional_text(row.get('street_address')),
            'postal_code': row['postal_code'],
            'footprint': _coerce_footprint(row.get('polygon_wkt')),
            'area': float(_area),
            'catchment_fj': max(CATCHMENT_FJ_FLOOR, float(_fj)),
        }

    return output
