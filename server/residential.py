import logging
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon
from shapely.ops import unary_union

try:
    import osmnx as ox

    OSMNX_AVAILABLE = True
except ImportError:
    ox = None
    OSMNX_AVAILABLE = False


LOGGER = logging.getLogger(__name__)

ArealGeometry = Union[Polygon, MultiPolygon]

RESIDENTIAL_BUILDING_VALUES = {
    "apartments",
    "bungalow",
    "cabin",
    "detached",
    "dormitory",
    "ger",
    "house",
    "houseboat",
    "residential",
    "semidetached_house",
    "static_caravan",
    "terrace",
    "trullo",
}

APARTMENT_LIKE_BUILDINGS = {
    "apartments",
    "dormitory",
    "residential",
}

AMBIGUOUS_BUILDING_VALUES = {
    "building",
    "yes",
}

NON_RESIDENTIAL_BUILDING_VALUES = {
    "barn",
    "carport",
    "cathedral",
    "chapel",
    "church",
    "civic",
    "commercial",
    "construction",
    "factory",
    "farm_auxiliary",
    "fire_station",
    "garage",
    "garages",
    "government",
    "grandstand",
    "greenhouse",
    "hangar",
    "hospital",
    "hotel",
    "industrial",
    "kiosk",
    "kindergarten",
    "mosque",
    "office",
    "parking",
    "pavilion",
    "public",
    "retail",
    "roof",
    "school",
    "service",
    "shed",
    "sports_centre",
    "stadium",
    "supermarket",
    "synagogue",
    "temple",
    "train_station",
    "transportation",
    "warehouse",
}

NON_RESIDENTIAL_SIGNAL_COLUMNS = (
    "aeroway",
    "amenity",
    "healthcare",
    "industrial",
    "landuse",
    "leisure",
    "man_made",
    "military",
    "office",
    "power",
    "public_transport",
    "railway",
    "shop",
    "tourism",
)

RESIDENTIAL_LANDUSE_VALUES = {
    "residential",
}


class ResidentialCache:
    """Sample realistic household locations inside residential features per CBG.

    The sampler prefers residential building footprints and falls back to
    residential landuse polygons. If OSM is unavailable or sparse, it samples
    within a slightly buffered interior of the CBG polygon.
    """

    def __init__(self, gdf, use_buildings: bool = True):
        self.use_buildings = bool(use_buildings)
        self._rng = random.Random()

        self.cbg_gdf = self._normalize_cbg_gdf(gdf)
        self.metric_crs = self._pick_metric_crs(self.cbg_gdf)
        self.cbg_metric = self.cbg_gdf.to_crs(self.metric_crs)
        self.to_wgs84 = Transformer.from_crs(
            self.metric_crs, "EPSG:4326", always_xy=True
        )

        self._fallback_geom_by_cbg: Dict[str, ArealGeometry] = {}
        self._building_candidates_by_cbg: Dict[
            str, List[Tuple[ArealGeometry, float]]
        ] = {}
        self._landuse_candidates_by_cbg: Dict[
            str, List[Tuple[ArealGeometry, float]]
        ] = {}

        self._build_candidate_cache()

    def sample_home_location(self, cbg: str) -> Tuple[Optional[float], Optional[float]]:
        cbg_id = self._normalize_cbg(cbg)
        if not cbg_id:
            return None, None

        candidates = []
        if self.use_buildings:
            candidates = self._building_candidates_by_cbg.get(cbg_id, [])
        if not candidates:
            candidates = self._landuse_candidates_by_cbg.get(cbg_id, [])

        if candidates:
            geom = self._weighted_choice(candidates)
            point = self._sample_point_in_geometry(geom)
        else:
            fallback_geom = self._fallback_geom_by_cbg.get(cbg_id)
            if fallback_geom is None or fallback_geom.is_empty:
                return None, None
            point = self._sample_point_in_geometry(fallback_geom)

        lon, lat = self.to_wgs84.transform(point.x, point.y)
        return float(lat), float(lon)

    def _build_candidate_cache(self) -> None:
        for _, row in self.cbg_metric.iterrows():
            cbg_id = self._normalize_cbg(row.get("CensusBlockGroup"))
            geom = self._clean_areal_geometry(row.geometry)
            if not cbg_id or geom is None:
                continue
            fallback = self._clean_areal_geometry(geom.buffer(-15))
            self._fallback_geom_by_cbg[cbg_id] = (
                fallback if fallback is not None and not fallback.is_empty else geom
            )

        osm_features = self._fetch_osm_features()
        if osm_features.empty:
            LOGGER.info("ResidentialCache: OSM residential features unavailable; using CBG fallback sampling")
            return

        osm_metric = osm_features.to_crs(self.metric_crs)
        building_features = self._filter_residential_buildings(osm_metric)
        landuse_features = self._filter_residential_landuse(osm_metric)

        for _, cbg_row in self.cbg_metric.iterrows():
            cbg_id = self._normalize_cbg(cbg_row.get("CensusBlockGroup"))
            cbg_geom = self._clean_areal_geometry(cbg_row.geometry)
            if not cbg_id or cbg_geom is None:
                continue

            if self.use_buildings and not building_features.empty:
                self._building_candidates_by_cbg[cbg_id] = self._clip_candidate_geometries(
                    building_features,
                    cbg_geom,
                    weight_builder=self._building_weight,
                )
            if not landuse_features.empty:
                self._landuse_candidates_by_cbg[cbg_id] = self._clip_candidate_geometries(
                    landuse_features,
                    cbg_geom,
                    weight_builder=self._landuse_weight,
                )

        total_buildings = sum(len(v) for v in self._building_candidates_by_cbg.values())
        total_landuse = sum(len(v) for v in self._landuse_candidates_by_cbg.values())
        LOGGER.info(
            "ResidentialCache prepared %s residential building candidates and %s landuse candidates across %s CBGs",
            total_buildings,
            total_landuse,
            len(self._fallback_geom_by_cbg),
        )

    def _fetch_osm_features(self) -> gpd.GeoDataFrame:
        if not OSMNX_AVAILABLE:
            LOGGER.info("ResidentialCache: osmnx not installed; skipping OSM residential lookup")
            return self._empty_geodataframe()

        min_lng, min_lat, max_lng, max_lat = self.cbg_gdf.total_bounds
        tags = {
            "building": True,
            "landuse": list(RESIDENTIAL_LANDUSE_VALUES),
            "residential": True,
        }

        try:
            ox.settings.use_cache = True
        except Exception:
            pass

        try:
            features = self._osmnx_features_from_bbox(
                north=max_lat,
                south=min_lat,
                east=max_lng,
                west=min_lng,
                tags=tags,
            )
        except Exception:
            LOGGER.warning(
                "ResidentialCache: failed to fetch OSM features for bbox %.6f,%.6f,%.6f,%.6f",
                min_lng,
                min_lat,
                max_lng,
                max_lat,
                exc_info=True,
            )
            return self._empty_geodataframe()

        if features is None or len(features) == 0:
            return self._empty_geodataframe()

        feature_gdf = gpd.GeoDataFrame(features).reset_index(drop=False)
        if "geometry" not in feature_gdf.columns:
            return self._empty_geodataframe()

        feature_gdf = feature_gdf[feature_gdf.geometry.notna()].copy()
        if feature_gdf.empty:
            return self._empty_geodataframe()

        if feature_gdf.crs is None:
            feature_gdf = feature_gdf.set_crs("EPSG:4326", allow_override=True)
        else:
            feature_gdf = feature_gdf.to_crs("EPSG:4326")

        feature_gdf = feature_gdf[
            feature_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ].copy()
        if feature_gdf.empty:
            return self._empty_geodataframe()

        cluster_union = unary_union(self.cbg_gdf.geometry)
        feature_gdf = feature_gdf[feature_gdf.geometry.intersects(cluster_union)].copy()
        return feature_gdf if not feature_gdf.empty else self._empty_geodataframe()

    @staticmethod
    def _osmnx_features_from_bbox(north: float, south: float, east: float, west: float, tags):
        callables = []
        if hasattr(ox, "features_from_bbox"):
            callables.append(ox.features_from_bbox)
        if hasattr(ox, "geometries_from_bbox"):
            callables.append(ox.geometries_from_bbox)

        for func in callables:
            try:
                return func(north, south, east, west, tags)
            except TypeError:
                try:
                    return func((north, south, east, west), tags)
                except TypeError:
                    continue

        raise RuntimeError("osmnx does not expose a compatible bbox feature query")

    def _filter_residential_buildings(self, feature_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if feature_gdf.empty:
            return feature_gdf
        mask = feature_gdf.apply(self._is_residential_building, axis=1)
        return feature_gdf[mask].copy()

    def _filter_residential_landuse(self, feature_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if feature_gdf.empty:
            return feature_gdf
        mask = feature_gdf.apply(self._is_residential_landuse, axis=1)
        return feature_gdf[mask].copy()

    def _clip_candidate_geometries(
        self,
        source_gdf: gpd.GeoDataFrame,
        cbg_geom: ArealGeometry,
        weight_builder,
    ) -> List[Tuple[ArealGeometry, float]]:
        if source_gdf.empty:
            return []

        try:
            candidate_idx = list(source_gdf.sindex.intersection(cbg_geom.bounds))
            subset = source_gdf.iloc[candidate_idx]
        except Exception:
            subset = source_gdf

        subset = subset[subset.geometry.intersects(cbg_geom)]
        if subset.empty:
            return []

        candidates: List[Tuple[ArealGeometry, float]] = []
        for _, row in subset.iterrows():
            clipped = self._clean_areal_geometry(row.geometry.intersection(cbg_geom))
            if clipped is None or clipped.is_empty:
                continue
            weight = weight_builder(row, clipped)
            if weight <= 0:
                continue
            candidates.append((clipped, weight))
        return candidates

    def _sample_point_in_geometry(self, geom: ArealGeometry) -> Point:
        target = self._clean_areal_geometry(geom)
        if target is None or target.is_empty:
            return geom.representative_point()

        if isinstance(target, MultiPolygon):
            parts = [part for part in target.geoms if not part.is_empty]
            if not parts:
                return target.representative_point()
            target = self._weighted_choice([(part, max(part.area, 1.0)) for part in parts])

        minx, miny, maxx, maxy = target.bounds
        for _ in range(40):
            point = Point(
                self._rng.uniform(minx, maxx),
                self._rng.uniform(miny, maxy),
            )
            if target.covers(point):
                return point

        return target.representative_point()

    def _building_weight(self, row, clipped_geom: ArealGeometry) -> float:
        building = self._string_value(row, "building")
        area = max(float(clipped_geom.area), 25.0)
        area = min(area, 2500.0)

        if building in APARTMENT_LIKE_BUILDINGS:
            multiplier = 6.0
        elif building in {"terrace", "semidetached_house"}:
            multiplier = 1.6
        elif building in RESIDENTIAL_BUILDING_VALUES:
            multiplier = 1.0
        elif building in AMBIGUOUS_BUILDING_VALUES:
            multiplier = 0.35
        else:
            multiplier = 0.2

        if self._string_value(row, "residential"):
            multiplier = max(multiplier, 1.5)

        return area * multiplier

    @staticmethod
    def _landuse_weight(row, clipped_geom: ArealGeometry) -> float:
        area = max(float(clipped_geom.area), 100.0)
        return min(area, 25000.0)

    def _weighted_choice(self, items: Sequence[Tuple[object, float]]):
        total = sum(weight for _, weight in items)
        if total <= 0:
            return items[0][0]

        target = self._rng.uniform(0, total)
        running = 0.0
        for item, weight in items:
            running += weight
            if running >= target:
                return item
        return items[-1][0]

    def _is_residential_building(self, row) -> bool:
        building = self._string_value(row, "building")
        if not building:
            return False

        if building in NON_RESIDENTIAL_BUILDING_VALUES:
            return False

        if building in RESIDENTIAL_BUILDING_VALUES:
            return True

        building_use = self._string_value(row, "building:use")
        if building_use in RESIDENTIAL_BUILDING_VALUES:
            return True

        if self._string_value(row, "residential"):
            return True

        if building in AMBIGUOUS_BUILDING_VALUES:
            for col in NON_RESIDENTIAL_SIGNAL_COLUMNS:
                if col == "landuse":
                    landuse = self._string_value(row, col)
                    if landuse and landuse not in RESIDENTIAL_LANDUSE_VALUES:
                        return False
                    continue
                if self._string_value(row, col):
                    return False
            return True

        return False

    def _is_residential_landuse(self, row) -> bool:
        landuse = self._string_value(row, "landuse")
        if landuse in RESIDENTIAL_LANDUSE_VALUES:
            return True
        return bool(self._string_value(row, "residential"))

    @staticmethod
    def _normalize_cbg_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if "CensusBlockGroup" not in gdf.columns and "GEOID" in gdf.columns:
            gdf = gdf.copy()
            gdf["CensusBlockGroup"] = gdf["GEOID"]
        if "GEOID" not in gdf.columns and "CensusBlockGroup" in gdf.columns:
            gdf = gdf.copy()
            gdf["GEOID"] = gdf["CensusBlockGroup"]

        for col in ("CensusBlockGroup", "GEOID"):
            if col in gdf.columns:
                gdf[col] = gdf[col].map(ResidentialCache._normalize_cbg)
        return gdf

    @staticmethod
    def _normalize_cbg(cbg) -> Optional[str]:
        if cbg is None or (isinstance(cbg, float) and pd.isna(cbg)):
            return None
        text = str(cbg).strip()
        if not text:
            return None
        if text.endswith(".0"):
            text = text[:-2]
        digits = "".join(ch for ch in text if ch.isdigit())
        return digits.zfill(12) if digits else None

    @classmethod
    def _normalize_cbg_gdf(cls, gdf) -> gpd.GeoDataFrame:
        if gdf is None:
            raise ValueError("ResidentialCache requires a GeoDataFrame")

        cbg_gdf = gpd.GeoDataFrame(gdf).copy()
        cbg_gdf = cls._normalize_cbg_columns(cbg_gdf)
        if "CensusBlockGroup" not in cbg_gdf.columns:
            raise ValueError("GeoDataFrame is missing CBG identifiers")
        if cbg_gdf.crs is None:
            cbg_gdf = cbg_gdf.set_crs("EPSG:4326", allow_override=True)
        else:
            cbg_gdf = cbg_gdf.to_crs("EPSG:4326")

        cbg_gdf = cbg_gdf[cbg_gdf.geometry.notna()].copy()
        cbg_gdf = cbg_gdf[
            cbg_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
        ].copy()
        if cbg_gdf.empty:
            raise ValueError("GeoDataFrame does not contain polygon CBG geometries")
        return cbg_gdf

    @staticmethod
    def _pick_metric_crs(gdf: gpd.GeoDataFrame) -> str:
        try:
            estimated = gdf.estimate_utm_crs()
            if estimated:
                return str(estimated)
        except Exception:
            pass
        return "EPSG:3857"

    @staticmethod
    def _string_value(row, key: str) -> Optional[str]:
        if key not in row:
            return None
        value = row.get(key)
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        text = str(value).strip().lower()
        return text or None

    @staticmethod
    def _clean_areal_geometry(geom):
        if geom is None or geom.is_empty:
            return None
        if isinstance(geom, (Polygon, MultiPolygon)):
            return geom
        if isinstance(geom, GeometryCollection):
            polygons = [
                part
                for part in geom.geoms
                if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty
            ]
            if not polygons:
                return None
            return unary_union(polygons)
        return None

    @staticmethod
    def _empty_geodataframe() -> gpd.GeoDataFrame:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
