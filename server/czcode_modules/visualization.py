import json

import folium
import pandas as pd
from folium import plugins

from common_geo import normalize_cbg

from .metrics import Helpers, cbg_population


class Visualizer:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def get_color_for_ratio(self, ratio):
        if ratio >= 0.8:
            return "#0000FF"
        elif ratio >= 0.6:
            return "#008000"
        elif ratio >= 0.4:
            return "#FFFF00"
        elif ratio >= 0.2:
            return "#FFA500"
        return "#FF0000"

    @staticmethod
    def cbg_geocode(cbg_id, gdf=None):
        try:
            point = gdf[gdf['CensusBlockGroup'] == normalize_cbg(cbg_id)].representative_point()
            center = point.iloc[0]
            return {
                'latitude': center.y,
                'longitude': center.x
            }
        except Exception:
            return {'latitude': None, 'longitude': None}

    def generate_maps(self, G, gdf, algorithm_result):
        def safe_center():
            try:
                seed = Visualizer.cbg_geocode(self.config.core_cbg, gdf)
                if seed['latitude'] is None or seed['longitude'] is None:
                    for cbg in algorithm_result[0]:
                        pos = Visualizer.cbg_geocode(cbg, gdf)
                        if pos['latitude'] is not None and pos['longitude'] is not None:
                            return [pos['latitude'], pos['longitude']]

                    return self.config.map["default_location"]

                return [seed['latitude'], seed['longitude']]
            except Exception:
                self.logger.warning("Error getting center coordinates, using default", exc_info=True)
                return self.config.map["default_location"]

        center = safe_center()
        self.map_obj = folium.Map(location=center, zoom_start=self.config.map["zoom_start"])
        features = []
        for i, cbg in enumerate(algorithm_result[0]):
            try:
                ratio = Helpers.calculate_cbg_ratio(G, cbg, algorithm_result[0])
                color = self.get_color_for_ratio(ratio)
                shape = gdf[gdf['CensusBlockGroup'] == cbg]
                if shape.empty:
                    continue
                shape = shape.to_crs("EPSG:4326")
                geojson = json.loads(shape.to_json())
                feature = geojson['features'][0]
                feature['properties']['times'] = [(pd.Timestamp('today') + pd.Timedelta(i, 'D')).isoformat()]
                feature['properties']['style'] = {'fillColor': color, 'color': color, 'fillOpacity': 0.7}
                features.append(feature)

                loc = shape.representative_point().iloc[0]
                folium.Marker(location=[loc.y, loc.x], popup=f'{cbg} - Population: {cbg_population(cbg, self.config, self.logger)}').add_to(self.map_obj)
            except Exception:
                self.logger.error(f"Error processing CBG {cbg} for map", exc_info=True)
        self.map_obj.add_child(plugins.TimestampedGeoJson(
            {'type': 'FeatureCollection', 'features': features},
            period='PT6H',
            add_last_point=True,
            auto_play=False,
            loop=False
        ))
        for cbg in self.config.black_cbgs:
            shape = gdf[gdf['CensusBlockGroup'] == cbg]
            if not shape.empty:
                shape = shape.to_crs("EPSG:4326")
                folium.GeoJson(
                    json.loads(shape.to_json()),
                    style_function=lambda x: {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.7}
                ).add_to(self.map_obj)
