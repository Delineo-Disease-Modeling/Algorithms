import importlib
import logging
import sys
from pathlib import Path

import networkx as nx


SERVER_DIR = Path(__file__).resolve().parents[2] / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

czcode = importlib.import_module("czcode")
analysis_service = importlib.import_module("server_app.analysis_service")


SEED_GUARD_DISTANCE_KM = 20.0
CBG_CENTERS = {
    "seed": (0.0, 0.0),
    "local1": (0.0, 0.05),
    "local2": (0.0, 0.06),
    "remote": (0.0, 0.30),
    "remote2": (0.0, 0.31),
}
CBG_POPULATIONS = {
    "seed": 100,
    "local1": 100,
    "local2": 100,
    "remote": 100,
    "remote2": 100,
}


class DummyConfig:
    def __init__(self, *args, **kwargs):
        pass


def _build_graph():
    graph = nx.Graph()
    graph.add_edge("seed", "local1", weight=10.0)
    graph.add_edge("seed", "remote", weight=9.0)
    graph.add_edge("local1", "local2", weight=8.0)
    graph.add_edge("remote", "remote2", weight=100.0)
    return graph


def _fake_population(cbg, _config, _logger):
    return CBG_POPULATIONS[cbg]


def test_greedy_weight_seed_guard_prevents_remote_branch_domination(monkeypatch):
    monkeypatch.setattr("czcode_modules.clustering.cbg_population", _fake_population)

    clustering = czcode.Clustering(DummyConfig(), logging.getLogger("test-seed-guard"))
    cluster, population = clustering.greedy_weight_seed_guard(
        _build_graph(),
        "seed",
        350,
        seed_guard_distance_km=SEED_GUARD_DISTANCE_KM,
        cbg_centers=CBG_CENTERS,
    )

    assert cluster == ["seed", "local1", "remote", "local2"]
    assert population == 400


def test_frontier_ranking_ignores_excluded_remote_movement(monkeypatch):
    monkeypatch.setattr(analysis_service, "Config", DummyConfig)
    monkeypatch.setattr(analysis_service, "setup_logging", lambda _config: logging.getLogger("test-frontier"))
    monkeypatch.setattr(analysis_service, "cbg_population", _fake_population)
    monkeypatch.setattr("czcode_modules.clustering.cbg_population", _fake_population)

    service = analysis_service.PreviewClusteringService(clustering_store=None)
    service.resources.get_cbg_centers = lambda *args, **kwargs: CBG_CENTERS

    candidates, missing = service.rank_frontier_candidates_for_cluster(
        graph=_build_graph(),
        seed_cbg="seed",
        cluster_cbgs=["seed", "local1", "remote"],
        algorithm="greedy_weight_seed_guard",
        seed_guard_params={"seed_guard_distance_km": SEED_GUARD_DISTANCE_KM},
    )

    assert missing == []
    assert [candidate["cbg"] for candidate in candidates] == ["local2", "remote2"]
    assert candidates[0]["score"] == 8.0
    assert candidates[1]["score"] == 0.0
    assert candidates[1]["movement_to_full_cluster"] == 100.0
    assert candidates[1]["movement_contributes_after_selection"] is False
