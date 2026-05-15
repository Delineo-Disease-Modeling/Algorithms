import logging

import pandas as pd

from czcode_modules.graph import GraphBuilder


def test_graph_builder_shares_record_parsing_for_graph_types():
    df = pd.DataFrame([
        {
            'poi_cbg': '240010001001',
            'visitor_daytime_cbgs': '{"240010001001": 2, "240010002002": 3, "bad": 4}',
        },
        {
            'poi_cbg': '240010001001',
            'visitor_daytime_cbgs': '{"240010002002": 7}',
        },
        {
            'poi_cbg': 'bad',
            'visitor_daytime_cbgs': '{"240010002002": 5}',
        },
        {
            'poi_cbg': '240010003003',
            'visitor_daytime_cbgs': 'not-json',
        },
    ])

    builder = GraphBuilder(logging.getLogger('test-graph-builder'))

    graph = builder.gen_graph(df)
    digraph = builder.gen_digraph(df)

    assert graph.nodes['240010001001']['self_weight'] == 2
    assert graph['240010001001']['240010002002']['weight'] == 10
    assert None not in graph

    assert digraph.nodes['240010001001']['self_weight'] == 2
    assert digraph['240010002002']['240010001001']['weight'] == 10
    assert None not in digraph
