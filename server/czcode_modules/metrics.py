import pandas as pd

from .cache_service import DEFAULT_ALGORITHM_CACHE


def cbg_population(cbg, config, logger, cache_service=DEFAULT_ALGORITHM_CACHE):
    try:
        population_df = cache_service.get_or_load_population(
            config.paths["population_csv"],
            lambda: pd.read_csv(
                config.paths["population_csv"],
                index_col='census_block_group',
                usecols=['census_block_group', 'B01003e1'],
            ),
        )
    except Exception as e:
        logger.error(f"Error loading population data: {e}")
        return 0

    try:
        cbg_int = int(float(cbg))
        if cbg_int in population_df.index:
            return int(population_df.loc[cbg_int].B01003e1)
        logger.warning(f"CBG {cbg} not found in population data")
        return 0
    except (ValueError, TypeError, KeyError) as e:
        logger.warning(f"Error retrieving population for CBG {cbg}: {e}")
        return 0


class Helpers:
    @staticmethod
    def calculate_cbg_ratio(G, cbg, cluster_cbgs):
        movement_in = 0
        movement_out = 0
        for neighbor in G.adj[cbg]:
            if neighbor in cluster_cbgs:
                movement_in += G.adj[cbg][neighbor]['weight'] / 2
            else:
                movement_out += G.adj[cbg][neighbor]['weight']
        total_movement = movement_in + movement_out
        return movement_in / total_movement if total_movement > 0 else 0

    @staticmethod
    def calculate_movement_stats(G, cluster_cbgs):
        movement_in = 0
        movement_out = 0
        missing_cbgs = []
        for cbg in cluster_cbgs:
            if cbg not in G:
                missing_cbgs.append(cbg)
                continue

            movement_in += float(G.nodes[cbg].get('self_weight', 0))
            for neighbor in G.adj[cbg]:
                if neighbor in cluster_cbgs:
                    movement_in += G.adj[cbg][neighbor]['weight'] / 2
                else:
                    movement_out += G.adj[cbg][neighbor]['weight']

        if missing_cbgs:
            missing_sorted = sorted(set(missing_cbgs))
            raise ValueError(
                f"CBGs not present in mobility graph: {', '.join(missing_sorted)}"
            )

        total = movement_in + movement_out
        return {'in': movement_in, 'out': movement_out, 'ratio': movement_in / total if total > 0 else 0}
