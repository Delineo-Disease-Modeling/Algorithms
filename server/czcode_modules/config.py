import os
from datetime import datetime as _datetime
from typing import List, Optional

from common_geo import STATE_FIPS_TO_ABBR, get_neighboring_states, normalize_cbg


class Config:
    def __init__(self, cbg, min_pop, patterns_file=None, patterns_folder=None,
                 month=None, start_date: Optional[_datetime] = None):
        cbg = normalize_cbg(cbg)

        fips = cbg[:2]
        abbr = STATE_FIPS_TO_ABBR.get(fips)
        self.states = [abbr] if abbr else []
        self.states = list(set(self.states) | set(get_neighboring_states(self.states)))

        self.location_name = f'{cbg}'
        self.core_cbg = cbg
        self.min_cluster_pop = min_pop
        self.output_dir = r"./output"
        self.start_date = start_date

        if month is None and start_date is not None:
            month = start_date.strftime('%Y-%m')

        self.patterns_folder = patterns_folder
        self.month = month
        resolved_patterns_csv = self._resolve_patterns_file(patterns_file, patterns_folder, month)

        self.paths = {
            "shapefiles_dir": r"./data/shapefiles/",
            "patterns_csv": resolved_patterns_csv,
            "population_csv": r"./data/cbg_b01.csv",
            "output_yaml": "cbg_info.yaml",
            "output_html": "map.html"
        }
        self.map = {
            "default_location": [0.0, 0.0],
            "zoom_start": 12
        }
        self.ratio_colors = {
            0.8: "#0000FF",
            0.6: "#008000",
            0.4: "#FFFF00",
            0.2: "#FFA500",
            0.0: "#FF0000",
        }
        self.black_cbgs = []
        os.makedirs(self.output_dir, exist_ok=True)

    def _resolve_patterns_file(self, patterns_file, patterns_folder, month):
        if patterns_file:
            if os.path.exists(patterns_file):
                return patterns_file
            raise FileNotFoundError(f"Patterns file not found: {patterns_file}")

        search_root = patterns_folder or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "patterns"
        )
        month_key = str(month or "").strip()
        state_candidates: List[str] = []

        if month_key:
            core = str(self.core_cbg or "").strip()
            if len(core) >= 2:
                state_hint = STATE_FIPS_TO_ABBR.get(core[:2])
                if state_hint:
                    state_candidates.append(state_hint)

            for state in self.states:
                state_code = str(state or "").strip().upper()
                if state_code and state_code not in state_candidates:
                    state_candidates.append(state_code)

            exts = ('.parquet', '.csv.gz', '.converted.csv', '.csv')
            for state in state_candidates:
                stem = f"{month_key}-{state}"
                for ext in exts:
                    path = os.path.join(search_root, state, f"{stem}{ext}")
                    if os.path.exists(path):
                        return path

            from patterns_loader import _available_months_for_state, closest_month
            for state in state_candidates:
                available = _available_months_for_state(state, search_root)
                nearest = closest_month(month_key, available)
                if nearest and nearest != month_key:
                    stem = f"{nearest}-{state}"
                    for ext in exts:
                        path = os.path.join(search_root, state, f"{stem}{ext}")
                        if os.path.exists(path):
                            import logging
                            logging.getLogger('cbg_clustering').info(
                                f"No data for {month_key}, using closest month: {nearest}"
                            )
                            return path

        states_msg = ", ".join(state_candidates) if month_key and state_candidates else "unknown state"
        raise FileNotFoundError(
            f"No patterns file found for month '{month_key}' "
            f"under {search_root} for {states_msg}. "
            "Expected files like '<DATA>/patterns/<STATE>/YYYY-MM-<STATE>.parquet'."
        )
