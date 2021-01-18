from collections import Counter
from itertools import repeat
from pathlib import Path
from typing import Tuple
from ortools.linear_solver import pywraplp
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import pandas as pd
import numpy as np
from ..document import Document
from ..alignment import Alignment
from ..amr import AMR


def expand_graph(graph: AMR, corpus: Document) -> None:
    """
    Expand the given `graph` by adding edges between all nodes in the same sentence,
    according to the provided `corpus`.
    This expansion is done **in place**.

    Parameters:
        graph (AMR): Graph to be expanded.
        corpus (Document): Corpus corresponding to the `graph`.
    """
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


def calculate_node_data(corpus: Document, alignment: Alignment) -> Tuple[Counter, dict, dict, dict]:
    """
    Compute the number of occurences of each node,
    along with their depth (distance to the root) in each sentence graph they occur.
    This also computes in which sentences each node occurs (their position in the given corpus).
    Finally, this uses the `alignment` to get the number of words (span length) aligned to each node.

    Parameters:
        corpus (Document): Corpus from which to extract the data.
        alignment (Alignment): Concept alignment data for the given `corpus`.

    Returns:
        tuple(Counter, dict, dict, dict): Number of occurences for each node. Their depths. Their positions and their span lengths. 
    """
    node_counts = Counter()
    node_depths = dict()
    node_positions = dict()
    span_lengths = dict()
    for i, doc in enumerate(corpus):
        doc_idx = alignment.get_sentence_position(doc.snt)
        doc_alignment = alignment[doc_idx] if doc_idx is not None else None
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

            # Get alignment info
            if doc_alignment is not None:
                if node_label.startswith('NER:'):
                    # Alignment for NER nodes is obtained from the first name (op1)
                    aligned_concept = node_label.split('.')[2]
                else:
                    aligned_concept = node_label

                try:
                    span_len = len(doc_alignment[aligned_concept])
                    span_lengths[node_label].append(span_len)
                except KeyError:
                    # No alignment for the concept found
                    span_lengths[node_label].append(0)
            else:
                # No alignment information for the sentence
                span_lengths[node_label].append(0)
    return node_counts, node_depths, node_positions, span_lengths


def get_node_features(amr: AMR, data: tuple) -> pd.DataFrame:
    """
    Create the local representations (features) for each node in the given graph.

    Parameters:
        amr (AMR): Graph from which to compute the attributes.
        data (tuple): Node data tuple created from the LiuEtAl2015.calculate_node_data() function.
    
    Returns:
        pd.DataFrame: Nodes local representations (features).
    """
    node_counts, node_depths, node_positions, span_lengths = data
    features_names = ['concept',
                      'n_freq_0', 'n_freq_1', 'n_freq_2', 'n_freq_5', 'n_freq_10',
                      'min_depth_1', 'min_depth_2', 'min_depth_3', 'min_depth_4', 'min_depth_5',
                      'avg_depth_1', 'avg_depth_2', 'avg_depth_3', 'avg_depth_4', 'avg_depth_5',
                      'n_fmst_pos_5', 'n_fmst_pos_6', 'n_fmst_pos_7', 'n_fmst_pos_10', 'n_fmst_pos_15',
                      'n_avg_pos_5', 'n_avg_pos_6', 'n_avg_pos_7', 'n_avg_pos_10', 'n_avg_pos_15',
                      'lngst_span_0', 'lngst_span_1', 'lngst_span_2', 'lngst_span_5', 'lngst_span_10',
                      'avg_span_0', 'avg_span_1', 'avg_span_2', 'avg_span_5', 'avg_span_10',
                      'ner', 'date',
                      'n_bias']
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
            if node == amr.get_top():
                # TOP node has depth 0
                depth = 0.0
            else:
                depth = 1.0 if min(node_depths[node_label]) >= t else 0.0
            features[node].append(depth)

        # avg_depth_1, avg_depth_2, avg_depth_3, avg_depth_4, avg_depth_5
        if node == amr.get_top():
            avg_depth = 0.0
        else:
            avg_depth = np.mean(node_depths[node_label])
        for t in [1, 2, 3, 4, 5]:
            depth = 1.0 if avg_depth >= t else 0.0
            features[node].append(depth)

        # fmst_pos_5, fmst_pos_6, fmst_pos_7, fmst_pos_10, fmst_pos_15
        for t in [5, 6, 7, 10, 15]:
            if node_label in node_positions:
                pos = 1.0 if min(node_positions[node_label]) >= t else 0.0
            else:
                # There is no information about this specific node
                pos = 0.0
            features[node].append(pos)

        # avg_pos_5, avg_pos_6, avg_pos_7, avg_pos_10, avg_pos_15
        if node_label in node_positions:
            avg_pos = np.mean(node_positions[node_label])
        else:
            avg_pos = 0.0
        for t in [5, 6, 7, 10, 15]:
            pos = 1.0 if avg_pos >= t else 0.0
            features[node].append(pos)

        # lngst_span_0, lngst_span_1, lngst_span_2, lngst_span_5, lngst_span_10
        for t in [0, 1, 2, 5, 10]:
            if node_label in span_lengths:
                span = 1.0 if max(span_lengths[node_label]) >= t else 0.0
            else:
                span = 0.0
            features[node].append(span)

        if node_label in span_lengths:
            avg_span = np.mean(span_lengths[node_label])
        else:
            avg_span = 0.0
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

    return pd.DataFrame(features,
                        index=features_names,
                        dtype=np.float32).T


def calculate_edge_data(corpus: Document) -> Tuple[Counter, dict]:
    """
    Compute the number of occurences of each edge and in which sentences they occur
    (their position in the given corpus).

    Parameters:
        corpus (Document): Corpus from which to extract the data.
    
    Returns:
        tuple(Counter, dict): Number of occurences of each edge and their positions.
    """
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


def get_edge_features(merged_graph: AMR, data: tuple, nodes_features: pd.DataFrame) -> pd.DataFrame:
    """
    Create the local representations (features) for each edge in the given graph.

    Parameters:
        merged_graph (AMR): Graph from which to compute the attributes.
        data (tuple): Edge data tuple created by the LiuEtAl2015.calculate_edge_data() function.
        nodes_features (pd.DataFrame): Node local representations (features) for the given `merged_graph`.
    
    Returns:
        pd.DataFrame: Edges local representations (features).
    """
    edge_counts, edge_positions = data
    features_names = ['label_1_05', 'label_1_066', 'label_1_075',
                      'label_2_05', 'label_2_066', 'label_2_075',
                      'e_freq_0', 'e_freq_1', 'e_freq_2', 'e_freq_5', 'e_freq_10',
                      'e_fmst_pos_5', 'e_fmst_pos_6', 'e_fmst_pos_7', 'e_fmst_pos_10', 'e_fmst_pos_15',
                      'e_avg_pos_5', 'e_avg_pos_6', 'e_avg_pos_7', 'e_avg_pos_10', 'e_avg_pos_15']
    node1_names = nodes_features.add_prefix(
        'node1_').columns[nodes_features.columns != 'bias']
    node2_names = nodes_features.add_prefix(
        'node2_').columns[nodes_features.columns != 'bias']
    features_names.extend(node1_names)
    features_names.extend(node2_names)
    features_names.extend(['expansion',
                           'exp_freq_0', 'exp_freq_1', 'exp_freq_2', 'exp_freq_5', 'exp_freq_10',
                           'e_bias'])

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


def ilp_optimisation(node_features: pd.DataFrame, edge_features: pd.DataFrame,
                     weights: np.array, top: str, nodes_cost: np.array = 0,
                     edge_cost: np.array = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run ILP optimization to select nodes and edges according to their features and the given weights.
    
    Parameters:
        node_features (pd.DataFrame): Node local representations (features).
        edge_features (pd.DataFrame): Edge local representations (features).
        weights (np.array): Feature weights to calculate a score for each node/edge.
        top (str): Which node (variable) to use as the root of the graph.
        nodes_cost (np.array): Value to sum into the computed score for each node
                               (`n` positions, `n` being the number of nodes).
        edges_cost (np.array): Value to sum into the computed score for each edge
                              (`e` positions, `e` being the number of nodes).
        
        Returns:
            tuple(pd.DataFrame, pd.DataFrame): Selected nodes and selected edges.
    """
    nodes_scores = np.dot(node_features, weights) + nodes_cost
    edge_scores = np.dot(edge_features, weights) + edge_cost

    solver = pywraplp.Solver('LiuEtAl2015',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    nodes_var = {n: solver.IntVar(0, 1, 'node[{}]'.format(n))
                 for n, _ in node_features.iterrows()}
    edges_var = {e: solver.IntVar(0, 1, 'edge[{}]'.format(e))
                 for e, _ in edge_features.iterrows()}
    flow_var = {(n1, n2): solver.IntVar(0, solver.Infinity(), 'flow[{}]'.format((n1, n2)))
                for n1, _ in node_features.iterrows()
                for n2, _ in node_features.iterrows()}

    # Constraints
    # If an edge is selected, both nodes have to be selected too
    for s, t in edges_var:
        edge_s_ct = solver.Constraint(
            0, 1, 'ct_edge[{}][{}]'.format((s, t), s))
        edge_s_ct.SetCoefficient(nodes_var[s], 1)
        edge_s_ct.SetCoefficient(edges_var[(s, t)], -1)

        if t != s:
            # Loops have only one constraint
            edge_t_ct = solver.Constraint(
                0, 1, 'ct_edge[{}][{}]'.format((s, t), t))
            edge_t_ct.SetCoefficient(nodes_var[t], 1)
            edge_t_ct.SetCoefficient(edges_var[(s, t)], -1)

        # Select at most one edge between two nodes
        # If there is more than one direction between the nodes
        if (t, s) in edges_var:
            self_loop_ct = solver.Constraint(
                0, 1, 'ct_self_loop[{}][{}]'.format(s, t))
            self_loop_ct.SetCoefficient(edges_var[(s, t)], 1)
            self_loop_ct.SetCoefficient(edges_var[(t, s)], 1)

    # Connectivity
    root_flow_ct = solver.Constraint(0, 0, 'root_flow_ct')
    for n, _ in node_features.iterrows():
        root_flow_ct.SetCoefficient(flow_var[(top, n)], 1)
        if n != top:
            root_flow_ct.SetCoefficient(nodes_var[n], -1)

            flow_consumption_ct = solver.Constraint(0, 0,
                                                    'flow_consumption[{}]'.format(n))
            for n2, _ in node_features.iterrows():
                # Incoming flow
                flow_consumption_ct.SetCoefficient(flow_var[(n2, n)], 1)
                # Outgoing flow
                if n2 != top:
                    flow_consumption_ct.SetCoefficient(flow_var[(n, n2)], -1)
            flow_consumption_ct.SetCoefficient(nodes_var[n], -1)

    # Flow must go through a selected edge
    for src, tgt in flow_var:
        if tgt != top:
            edge_flow_ct = solver.Constraint(0, solver.infinity(),
                                             'edge_flow[{}]'.format((src, tgt)))
            if (src, tgt) in edges_var:
                edge_flow_ct.SetCoefficient(
                    edges_var[(src, tgt)], nodes_scores.shape[0])
            edge_flow_ct.SetCoefficient(flow_var[(src, tgt)], -1)

    # Force tree structure
    tree_ct = dict()
    for n, _ in node_features.iterrows():
        tree_ct[n] = solver.Constraint(0, 1, 'tree[{}]'.format(n))

    for (s, t), _ in edge_features.iterrows():
        tree_ct[t].SetCoefficient(edges_var[(s, t)], 1)

    # Objective
    obj = solver.Objective()
    obj.SetMaximization()

    for i, v in enumerate(nodes_var):
        obj.SetCoefficient(nodes_var[v], nodes_scores[i])
    for i, v in enumerate(edges_var):
        obj.SetCoefficient(edges_var[v], edge_scores[i])

    solver.Solve()

    nodes = [True if nodes_var[n].solution_value() == 1.0 else False
             for n, _ in node_features.iterrows()]
    edges = [True if edges_var[e].solution_value() == 1.0 else False
             for e, _ in edge_features.iterrows()]
    return node_features.loc[nodes, :], edge_features.loc[edges, :]


def graph_local_representations(graph: AMR, node_data: tuple, edge_data: tuple) -> pd.DataFrame:
    """
    Concatenate the local representations of all nodes and edges in the given graph.

    Parameters:
        graph (AMR): Graph to which create the representations.
        node_data (tuple): Node data tuple created by the LiuEtAl2015.calculate_node_data() function.
        edge_data (tuple): Edge data tuple created by the LiuEtAl2015.calculate_edge_data() function.
    
    Returns:
        pd.DataFrame: Matrix containing all features for each node and edge in the `graph`.
    """
    nodes_features = get_node_features(graph, node_data)
    edge_features = get_edge_features(graph, edge_data, nodes_features)
    return pd.concat([nodes_features, edge_features], axis=0).fillna(0.0)


def prepare_training_data(training_path, gold_path, alignment):
    """
    Create the training instances, one for each node/edge in each graph in both training_path and gold_path.
    Both training and gold paths must be aligned, so that the `gold_path` documents corresponds to
    the target value (the summary) of the `training_path` document.

    Parameters:
        training_path (Path): Training document.
        gold_path (Path): Target/summary document.
        alignment (Alignment): Concept alignments containing information of both train and target documents.
    
    Returns:
        DataFrame: Matrix containing the attributes representations for each node/edge in both training and gold documents.
    """
    training_corpus = Document.read(training_path)
    training_graph = training_corpus.merge_graphs(collapse_ner=True,
                                                  collapse_date=True)
    node_data = calculate_node_data(training_corpus, alignment)
    edge_data = calculate_edge_data(training_corpus)

    summary_corpus = Document.read(gold_path)
    gold_summary_graph = summary_corpus.merge_graphs(collapse_ner=True,
                                                     collapse_date=True)
    sum_repr = graph_local_representations(gold_summary_graph,
                                           node_data, edge_data)
    sum_repr['name'] = training_path.stem
    sum_repr['type'] = 'target'

    train_repr = graph_local_representations(training_graph,
                                             node_data, edge_data)
    train_repr['name'] = training_path.stem
    train_repr['type'] = 'train'
    train_repr.at[training_graph.get_top(), 'top'] = True

    final_reprs = pd.concat([train_repr, sum_repr])
    final_reprs['concept'] = final_reprs['concept'].replace(0.0, np.nan)
    return final_reprs


def update_weights(weights: np.array, train: pd.DataFrame, gold: pd.DataFrame,
                   top: str, loss: str = 'perceptron') -> Tuple[np.array, pd.DataFrame, pd.DataFrame]:
    """
    Update a given array of weights via AdaGrad upon a given train-gold graph pair.
    The given loss function (perceptron or ramp) is used to compute how the array
    should be updated.

    Parameters:
        weights (np.array): Initial weight array.
        train (pd.DataFrame): Training instance (all local representations, nodes and edges,
                              of a single merged AMR graph for the texts beign summarized)
        gold (pd.DataFrame): Target instance (all local representations, nodes and edges,
                             of a single merged AMR graph for the gold summary)
        top (str): TOP node (variable) for the train graph.
        loss (str): Which loss function to use (perceptron or ramp).
    
    Returns:
        tuple(np.array, pd.DataFrame, pd.DataFrame): Triple with the updated weights, along with the selected edges
               (via ILP using the given weights) for both gold and train graphs,
               respectively (in the case of using a ramp loss function, otherwise None).
    """
    train_nodes = train.loc[train['n_bias'] == 1.0, :]
    train_edges = train.loc[train['e_bias'] == 1.0, :]

    if loss == 'perceptron':
        gold_global = gold.sum(axis=0)

        ilp_n, ilp_e = ilp_optimisation(train_nodes, train_edges, weights, top)

        ilp_global = ilp_n.sum(axis=0) + ilp_e.sum(axis=0)

        gold_n, gold_e = None, None
    elif loss == 'ramp':
        # Set to 1 all nodes/edges that are in training, but not in target
        cost_idx = train.index.difference(gold.index)
        cost = pd.Series(0.0, index=train.index)
        cost.loc[cost_idx] = 1.0

        # Run with negative costs
        gold_n, gold_e = ilp_optimisation(train_nodes, train_edges, weights, top,
                                          nodes_cost=-cost.loc[train_nodes.index],
                                          edge_cost=-cost.loc[train_edges.index])
        gold_global = gold_n.sum(axis=0) + gold_e.sum(axis=0)

        # Run with positive cost
        ilp_n, ilp_e = ilp_optimisation(train_nodes, train_edges, weights, top,
                                        nodes_cost=cost.loc[train_nodes.index],
                                        edge_cost=cost.loc[train_edges.index])
        ilp_global = ilp_n.sum(axis=0) + ilp_e.sum(axis=0)

    # Adagrad
    gradient = ilp_global - gold_global
    eta = 1.0
    epsilon = 1.0
    learning_rate = eta / np.sqrt(np.sum(gradient ** 2) + epsilon)
    new_weights = weights - learning_rate * gradient
    return new_weights, gold_e, ilp_e


def train(training_path: Path, gold_path: Path, alignment: Alignment, loss: str) -> np.array:
    """
    Train the weights for the scoring method using ILP and AdaGrad. The preprocessing is done parallelly.
    Each node/edge from each AMR graph is represented as a set of binary attributes.
    The importance score of a node/edge is given by the linear combination of its attributes given a weight vector.
    The weight vector is initialized as a vector of 1, then it is updated via AdaGrad
    using a loss function given as a parameter (perceptron or ramp) through supervised learning.

    Parameters:
        training_path (Path): The corpus to use as training.
        gold_path (Path): The corpus to use as target.
        alignment (Alignment): The concept alignments for both train and target corpora.
        loss (str): Which loss function to use (perceptron or ramp)
    
    Returns:
        array: Optimized weights for the scoring of nodes and edges.
    """
    # Create training instances parallelly through the prepare_training_data functiontances
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        # Organize arguments for mapping
        train_filepaths = list()
        target_filepaths = list()
        for instance_path in training_path.iterdir():
            train_filepaths.append(instance_path)
            target_filepaths.append(gold_path / instance_path.name)
        alignment_arg = repeat(alignment)

        # Create training and target representations
        result = executor.map(prepare_training_data,
                              train_filepaths,
                              target_filepaths,
                              alignment_arg)

    # Combine all results from the parallel processing
    # Also provide one-hot encoding for concept attributes
    local_reprs_df = pd.get_dummies(pd.concat(result),
                                    columns=['concept',
                                             'node1_concept',
                                             'node2_concept'],
                                    dtype=np.float32)
    pairs_groups = local_reprs_df.groupby(by='name')

    # Training
    # Name, type and top columns are going to be dropped, so we need to subtract 3 dimensions
    weights = np.ones(local_reprs_df.shape[1] - 3)

    for g in pairs_groups.groups:
        # Separate train and target through the type column and ignore non-feature columns
        instance_df = pairs_groups.get_group(g).drop(columns='name')
        top_train = instance_df.loc[instance_df['top'] == True].index[0]
        train = instance_df.query("type == 'train'").drop(
            columns=['type', 'top'])
        target = instance_df.query("type == 'target'").drop(
            columns=['type', 'top'])

        # Update weights
        weights, _, _ = update_weights(weights,
                                       train,
                                       target,
                                       top_train,
                                       loss=loss)
    return weights


def run(corpus: Document, alignment: Alignment, **kwargs: dict) -> AMR:
    """
    Run method.

    Parameters:
        corpus(Document): The corpus upon which the summarization process will be applied.
        alignment(Alignment): Concept alignments corresponding to the `corpus`.

    Returns:
        AMR: Summary graph created from the `corpus`.
    """
    training_path = kwargs.get('training')
    gold_path = kwargs.get('target')
    output_path = kwargs.get('output')
    weights_path = kwargs.get('model')
    loss = kwargs.get('loss')

    # Check arguments
    if not weights_path and not (training_path and gold_path):
        raise ValueError('LiuEtAl2015 method requires either training and '
                         'target arguments or pre-trained weights')

    # Train or load weights
    if not weights_path and (training_path and gold_path):
        weights = train(training_path, gold_path, alignment, loss)
        weights.to_csv(output_path / 'weights.csv')
    elif weights_path:
        weights = pd.read_csv(weights_path, index_col=0, squeeze=True)

    # Test
    if corpus:
        merged_test_graph = corpus.merge_graphs(collapse_ner=True,
                                                collapse_date=True)
        # expand_graph(merged_graph, corpus)
        test_node_data = calculate_node_data(corpus, alignment)
        test_edge_data = calculate_edge_data(corpus)
        test_repr = graph_local_representations(merged_test_graph,
                                                test_node_data,
                                                test_edge_data)
        test_repr = pd.get_dummies(test_repr,
                                   columns=['concept',
                                            'node1_concept',
                                            'node2_concept'],
                                   dtype=np.float32)
        test_repr = test_repr.reindex(columns=weights.index,
                                      fill_value=0.0)
        test_nodes = test_repr.loc[test_repr['n_bias'] == 1.0, :]
        test_edges = test_repr.loc[test_repr['e_bias'] == 1.0, :]
        _, sum_edges = ilp_optimisation(test_nodes, test_edges,
                                        weights,
                                        merged_test_graph.get_top())

        selected_edges = list()
        for s, t in sum_edges.index:
            for l in merged_test_graph.get_edge_data(s, t):
                selected_edges.append((s, t, l))
        sum_subgraph = merged_test_graph.edge_subgraph(selected_edges).copy()
        sum_subgraph.uncollapse_ner_nodes()
        sum_subgraph.uncollapse_date_nodes()

        return sum_subgraph
