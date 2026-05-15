from .clustering_utils import ClusteringUtilsMixin
from .directed_algorithms import DirectedClusteringMixin
from .greedy_algorithms import GreedyClusteringMixin
from .mobility_prune import MobilityPruneMixin
from .optimal_czi import OptimalCziMixin


class Clustering(
    GreedyClusteringMixin,
    DirectedClusteringMixin,
    MobilityPruneMixin,
    OptimalCziMixin,
    ClusteringUtilsMixin,
):
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
