import os

import yaml

from .metrics import cbg_population


class Exporter:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

    def generate_yaml_output(self, G, algorithm_result):
        cbg_info_list = []
        for cbg in algorithm_result[0]:
            try:
                cbg_str = str(cbg)
                pop_est = cbg_population(cbg, self.config, self.logger)
                movement_in_S, movement_out_S = 0, 0
                for neighbor in G.adj[cbg]:
                    if neighbor in algorithm_result[0]:
                        movement_in_S += G.adj[cbg][neighbor]['weight'] / 2
                    else:
                        movement_out_S += G.adj[cbg][neighbor]['weight']
                total_movement = movement_in_S + movement_out_S
                ratio = movement_in_S / total_movement if total_movement > 0 else None
                cbg_info_list.append({
                    "GEOID10": cbg_str,
                    "movement_in": movement_in_S,
                    "movement_out": movement_out_S,
                    "ratio": ratio,
                    "estimated_population": pop_est
                })
            except Exception:
                self.logger.error(f"Error processing CBG {cbg} for YAML output", exc_info=True)

        output_yaml_path = os.path.join(self.config.output_dir, self.config.paths["output_yaml"])
        with open(output_yaml_path, "w", encoding="utf-8") as outfile:
            yaml.dump(cbg_info_list, outfile)
        self.logger.info(f"YAML output saved to {output_yaml_path}")
