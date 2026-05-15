import networkx as nx

from server_app.frontier_candidates import FrontierCandidateAnalyzer


class EmptyResources:
    def get_cbg_centers(self, *args, **kwargs):
        return {}


def test_frontier_candidates_normalizes_candidate_ids(monkeypatch):
    graph = nx.Graph()
    graph.add_node('seed', self_weight=0)
    graph.add_node('candidate', self_weight=0)
    graph.add_edge('seed', 'candidate', weight=12)

    monkeypatch.setattr(
        'server_app.frontier_candidates.cbg_population',
        lambda cbg, config, logger: 100,
    )

    analyzer = FrontierCandidateAnalyzer(EmptyResources())
    candidates, missing_cluster_cbgs = analyzer.rank_frontier_candidates_for_cluster(
        graph=graph,
        seed_cbg='seed',
        cluster_cbgs=['seed'],
        algorithm='greedy_weight',
        limit=1,
    )

    assert missing_cluster_cbgs == []
    assert candidates == [
        {
            'cbg': 'candidate',
            'population': 100,
            'score': 12.0,
            'movement_to_cluster': 12.0,
            'movement_to_outside': 0.0,
            'rank': 1,
            'selected': False,
        }
    ]
