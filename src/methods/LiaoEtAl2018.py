from ..document import Document
from ..alignment import Alignment
from collections import Counter
from itertools import combinations
import pandas as pd
import numpy as np

corpus = Document.read(
    '..\\Corpora\\OpiSums-PT\\Textos_AMR\\O-Apanhador-no-Campo-de-Centeio\\O-Apanhador-no-Campo-de-Centeio.parsed')
alignments = Alignment.read_jamr('..\\corpora\\AMR-PT-OP\\SPAN-MANUAL\\combined_manual_jamr.txt')


def expand_graph(graph, corpus):
    for _, _, amr in corpus:
        for node1 in amr.nodes:
            for node2 in amr.nodes:
                u = graph.get_label_node(amr.get_node_label(node1))
                v = graph.get_label_node(amr.get_node_label(node2))

                if u == v or (node1, node2) in amr.edges():
                    # Don't expand if the edge already exists
                    continue
                elif (u, v) not in graph.edges():
                    graph.add_edge(u, v, key='expansion', count=1)
                else:
                    try:
                        # Edge has already been expanded
                        graph.edges[(u, v, 'expansion')]['count'] += 1
                    except KeyError:
                        # The edge exists but it is not an expansion
                        continue


def calculate_node_data(corpus, alignment):
    node_counts = Counter()
    node_depths = dict()
    node_positions = dict()
    span_lengths = dict()
    for i, doc in enumerate(corpus):
        doc_alignment = alignment[alignment.get_sentence_position(doc.snt)]
        for node in doc.amr.nodes():
            try:
                node_label = doc.amr.nodes[node]['label']
            except KeyError:
                node_label = node

            node_counts[node_label] += 1

            if node_label not in node_depths:
                node_depths[node_label] = list()
            node_depths[node_label].append(doc.amr.get_node_depth(node))

            if node_label not in node_positions:
                node_positions[node_label] = list()
            node_positions[node_label].append(i+1)

            if node_label not in span_lengths:
                span_lengths[node_label] = list()

            # Alignment for NER nodes is obtained from the first name (op1)
            aligned_concept = node_label.split('.')[2] if node_label.startswith('NER:') else node_label
            span_lengths[node_label].append(len(doc_alignment[aligned_concept]))
    return node_counts, node_depths, node_positions, span_lengths


def get_node_features(amr, data):
    node_counts, node_depths, node_positions, span_lengths = data
    features_names = ['concept',
                      'freq_0', 'freq_1', 'freq_2', 'freq_5', 'freq_10',
                      'min_depth_1', 'min_depth_2', 'min_depth_3', 'min_depth_4', 'min_depth_5',
                      'avg_depth_1', 'avg_depth_2', 'avg_depth_3', 'avg_depth_4', 'avg_depth_5',
                      'fmst_pos_5', 'fmst_pos_6', 'fmst_pos_7', 'fmst_pos_10', 'fmst_pos_15',
                      'avg_pos_5', 'avg_pos_6', 'avg_pos_7', 'avg_pos_10', 'avg_pos_15',
                      'lngst_span_0', 'lngst_span_1', 'lngst_span_2', 'lngst_span_5', 'lngst_span_10',
                      'avg_span_0', 'avg_span_1', 'avg_span_2', 'avg_span_5', 'avg_span_10',
                      'ner', 'date',
                      'bias']
    features = dict()

    for node in amr.nodes():
        features[node] = list()

        try:
            # Concept
            node_label = amr.nodes[node]['label']
        except KeyError:
            # Constant
            node_label = node

        # Concept feature
        features[node].append(node_label)

        # freq_0
        freq = 1.0 if node_counts[node_label] == 0 else 0.0
        features[node].append(freq)

        # freq_1, freq_2, freq_5, freq_10
        for t in [1, 2, 5, 10]:
            freq = 1.0 if node_counts[node_label] >= t else 0.0
            features[node].append(freq)

        # min_depth_1, min_depth_2, min_depth_3, min_depth_4, min_depth_5
        for t in [1, 2, 3, 4, 5]:
            depth = 1.0 if min(node_depths[node_label]) >= t else 0.0
            features[node].append(depth)

        # avg_depth_1, avg_depth_2, avg_depth_3, avg_depth_4, avg_depth_5
        for t in [1, 2, 3, 4, 5]:
            avg_depth = sum(node_depths[node_label]) * \
                1.0/len(node_depths[node_label])
            depth = 1.0 if avg_depth >= t else 0.0
            features[node].append(depth)

        # fmst_pos_5, fmst_pos_6, fmst_pos_7, fmst_pos_10, fmst_pos_15
        for t in [5, 6, 7, 10, 15]:
            pos = 1.0 if min(node_positions[node_label]) >= t else 0.0
            features[node].append(pos)

        # avg_pos_5, avg_pos_6, avg_pos_7, avg_pos_10, avg_pos_15
        for t in [5, 6, 7, 10, 15]:
            avg_pos = sum(node_positions[node_label]) * \
                1.0/len(node_positions[node_label])
            pos = 1.0 if avg_pos >= t else 0.0
            features[node].append(pos)
        
        # lngst_span_0, lngst_span_1, lngst_span_2, lngst_span_5, lngst_span_10
        for t in [0, 1, 2, 5, 10]:
            span = 1.0 if max(span_lengths[node_label]) >= t else 0.0
            features[node].append(span)

        avg_span = np.mean(span_lengths[node_label])
        for t in [0, 1, 2, 5, 10]:
            span = 1.0 if avg_span >= t else 0.0
            features[node].append(span)

        # ner
        ner = 1.0 if node_label.startswith('NER:') else 0.0
        features[node].append(ner)

        # date
        date = 1.0 if node_label.startswith('DATE:') else 0.0
        features[node].append(date)

        # bias
        features[node].append(1.0)

    return pd.get_dummies(pd.DataFrame(features,
                                       index=features_names,
                                       dtype=np.float32).T,
                          columns=['concept'],
                          dtype=np.float32)


def calculate_edge_data(corpus):
    edge_counts = Counter()
    edge_positions = dict()
    for i, doc in enumerate(corpus):
        for u, v, r in doc.amr.edges:
            try:
                u_label = doc.amr.nodes[u]['label']
            except KeyError:
                u_label = u

            try:
                v_label = doc.amr.nodes[v]['label']
            except KeyError:
                v_label = v

            edge_counts[u_label, v_label, r] += 1

            if (u_label, v_label) not in edge_positions:
                edge_positions[(u_label, v_label)] = list()
            edge_positions[(u_label, v_label)].append(i+1)
    return edge_counts, edge_positions


def get_edge_features(merged_graph, data, nodes_features):
    edge_counts, edge_positions = data
    features_names = ['label_1_05', 'label_1_066', 'label_1_075',
                      'label_2_05', 'label_2_066', 'label_2_075',
                      'freq_0', 'freq_1', 'freq_2', 'freq_5', 'freq_10',
                      'fmst_pos_5', 'fmst_pos_6', 'fmst_pos_7', 'fmst_pos_10', 'fmst_pos_15',
                      'avg_pos_5', 'avg_pos_6', 'avg_pos_7', 'avg_pos_10', 'avg_pos_15']
    node1_names = nodes_features.add_prefix(
        'node1_').columns[nodes_features.columns != 'bias']
    node2_names = nodes_features.add_prefix(
        'node2_').columns[nodes_features.columns != 'bias']
    features_names.extend(node1_names)
    features_names.extend(node2_names)
    features_names.extend(['expansion',
                           'exp_freq_0', 'exp_freq_1', 'exp_freq_2', 'exp_freq_5', 'exp_freq_10',
                            'bias'])

    # Get all edges between each pair of nodes
    edges = dict()
    for u, v, r in merged_graph.edges:
        if (u, v) not in edges:
            edges[(u, v)] = list()
        edges[(u, v)].append(r)

    features = dict()
    for u, v in edges:
        features[(u, v)] = list()
        u_label = merged_graph.get_node_label(u)
        v_label = merged_graph.get_node_label(v)

        # label
        l_freqs = Counter({l: edge_counts[(u_label, v_label, l)]
                           for l in edges[(u, v)]})
        frequent_labels = l_freqs.most_common(2)

        # label_1_05, label_1_066, label_1_075
        relative_freq = frequent_labels[0][1] * 1.0 / len(edges[(u, v)])
        for t in [0.5, 0.66, 0.75]:
            label = 1.0 if relative_freq >= t else 0.0
            features[(u, v)].append(label)

        # label_2_05, label_2_066, label_2_075
        if len(frequent_labels) > 1:
            relative_freq = frequent_labels[1][1] * 1.0 / len(edges[(u, v)])
            for t in [0.5, 0.66, 0.75]:
                label = 1.0 if relative_freq >= t else 0.0
                features[(u, v)].append(label)
        else:
            features[(u, v)].extend(3*[0])

        non_expanded_edges = [e for e in edges[(u, v)] if e != 'expansion']
        # freq_0
        freq = 1.0 if len(non_expanded_edges) == 0 else 0.0
        features[(u, v)].append(freq)
        # freq_1, freq_2, freq_5, freq_10
        for t in [1, 2, 5, 10]:
            freq = 1.0 if len(non_expanded_edges) >= t else 0.0
            features[(u, v)].append(freq)

        try:
            positions = edge_positions[(u_label, v_label)]
        except KeyError:
            positions = [0.0]

        # fmst_pos_5, fmst_pos_6, fmst_pos_7, fmst_pos_10, fmst_pos_15
        fmst_pos = min(positions)
        for t in [5, 6, 7, 10, 15]:
            pos = 1.0 if fmst_pos >= t else 0.0
            features[(u, v)].append(pos)

        # avg_pos_5, avg_pos_6, avg_pos_7, avg_pos_10, avg_pos_15
        avg_pos = np.mean(positions)
        for t in [5, 6, 7, 10, 15]:
            pos = 1.0 if avg_pos >= t else 0.0
            features[(u, v)].append(pos)

        # nodes features
        node1_features = nodes_features.loc[u,
                                            nodes_features.columns != 'bias']
        features[(u, v)].extend(node1_features)
        node2_features = nodes_features.loc[v,
                                            nodes_features.columns != 'bias']
        features[(u, v)].extend(node2_features)

        # expansion
        expansion = 1.0 if 'expansion' in edges[(u, v)] else 0.0
        features[(u, v)].append(expansion)

        # exp_freq_0
        freq = 1.0 if len(edges[(u, v)]) == 0 else 0.0
        features[(u, v)].append(freq)
        # exp_freq_1, exp_freq_2, exp_freq_5, exp_freq_10
        for t in [1, 2, 5, 10]:
            freq = 1.0 if len(edges[(u, v)]) >= t else 0.0
            features[(u, v)].append(freq)

        # bias
        features[(u, v)].append(1.0)

    return pd.DataFrame(features, index=features_names, dtype=np.float32).T


merged_graph = corpus.merge_graphs(collapse_ner=True, collapse_date=True)
expand_graph(merged_graph, corpus)
nodes_features = get_node_features(merged_graph,
                                   calculate_node_data(corpus, alignments))
edge_features = get_edge_features(merged_graph,
                                  calculate_edge_data(corpus),
                                  nodes_features)
