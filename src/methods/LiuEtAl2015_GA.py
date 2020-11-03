import os
import multiprocessing
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
from .LiuEtAl2015 import calculate_node_data, calculate_edge_data, graph_local_representations, ilp_optimisation, prepare_training_data


def get_edge_concepts(df):
    node1_columns = df.columns.str.startswith('node1_concept')
    node2_columns = df.columns.str.startswith('node2_concept')

    df_nodes1 = df.loc[:, node1_columns]
    df_nodes1.columns = df_nodes1.columns.str.replace('^node1_concept_', '',
                                                      regex=True)
    node1_concepts = df_nodes1[df_nodes1 ==
                               1].stack().reset_index(level=-1).drop(0, 1)
    node1_concepts.columns = ['node1']

    df_nodes2 = df.loc[:, node2_columns]
    df_nodes2.columns = df_nodes2.columns.str.replace('^node2_concept_', '',
                                                      regex=True)
    node2_concepts = df_nodes2[df_nodes2 ==
                               1].stack().reset_index(level=-1).drop(0, 1)
    node2_concepts.columns = ['node2']

    return node1_concepts.join(node2_concepts)


def calculate_edge_f_score(test_edges, gold_edges):
    test_concepts = get_edge_concepts(test_edges)
    gold_concepts = get_edge_concepts(gold_edges)

    test_tuples = pd.Series(
        zip(test_concepts['node1'], test_concepts['node2']))
    gold_tuples = pd.Series(
        zip(gold_concepts['node1'], gold_concepts['node2']))
    test_counts = test_tuples.isin(gold_tuples).value_counts()
    gold_counts = gold_tuples.isin(test_tuples).value_counts()

    true_positive = test_counts[True]
    false_positive = test_counts[False]
    false_negative = gold_counts[False]

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f_score = 2 * ((precision * recall) / (precision + recall))
    return f_score


def update_weights(weights):
    return np.random.rand(weights.shape[0])


def train(training_path, gold_path, alignment):
    # Create training instances
    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() - 1) as executor:
        # Organize arguments for mapping
        train_filepaths = list()
        target_filepaths = list()
        for instance_name in os.listdir(training_path):
            train_filepaths.append(os.path.join(training_path, instance_name))
            target_filepaths.append(os.path.join(gold_path, instance_name))
            break
        alignment_arg = repeat(alignment)

        # Create training and target representations
        result = executor.map(prepare_training_data,
                              train_filepaths,
                              target_filepaths,
                              alignment_arg)

    local_reprs_df = pd.get_dummies(pd.concat(result),
                                    columns=['concept',
                                             'node1_concept',
                                             'node2_concept'],
                                    dtype=np.float32)
    pairs_groups = local_reprs_df.groupby(by='name')

    weights = np.random.rand(local_reprs_df.shape[1] - 3)
    old_f = 0
    cur_f = 0

    while True:
        f_sum = 0
        n = 0
        for g in pairs_groups.groups:
            # Separate train and target through the type column and ignore non-feature columns
            instance_df = pairs_groups.get_group(g).drop(columns='name')
            top_train = instance_df.loc[instance_df['top'] == True].index[0]
            train = instance_df.query("type == 'train'").drop(
                columns=['type', 'top'])
            target = instance_df.query("type == 'target'").drop(
                columns=['type', 'top'])

            train_nodes = train.loc[train['n_bias'] == 1.0, :]
            train_edges = train.loc[train['e_bias'] == 1.0, :]
            ilp_n, ilp_e = ilp_optimisation(
                train_nodes, train_edges, weights, top_train)

            f_sum += calculate_edge_f_score(ilp_e, target.loc[target['e_bias'] == 1.0, :])
            n += 1

        old_f = cur_f
        cur_f = f_sum / n
        print(cur_f)
        if cur_f < old_f:
            break
        else:
            weights = update_weights(weights)


def run(corpus, alignment, **kwargs):
    training_path = kwargs.get('training')
    gold_path = kwargs.get('target')
    output_path = kwargs.get('output')
    weights_path = kwargs.get('weights')

    if not weights_path and not (training_path and gold_path):
        raise ValueError('LiuEtAl2015_GA method requires either training and '
                         'target arguments or pre-trained weights')

    if not weights_path and (training_path and gold_path):
        weights = train(training_path, gold_path, alignment)
        # weights.to_csv(os.path.join(output_path, 'weights.csv'))
    elif weights_path:
        weights = pd.read_csv(weights_path, index_col=0, squeeze=True)

    if corpus:
        merged_test_graph = corpus.merge_graphs(collapse_ner=True,
                                                collapse_date=True)
        test_node_data = calculate_node_data(corpus, alignment)
        test_edge_data = calculate_edge_data(corpus)
        test_repr = graph_local_representations(merged_test_graph,
                                                test_node_data,
                                                test_edge_data)

        # Run test
        test_repr = pd.get_dummies(test_repr,
                                   columns=['concept',
                                            'node1_concept',
                                            'node2_concept'],
                                   dtype=np.float32)
        test_repr = test_repr.reindex(columns=weights.index,
                                      fill_value=0.0)
        test_nodes = test_repr.loc[test_repr['n_bias'] == 1.0, :]
        test_edges = test_repr.loc[test_repr['e_bias'] == 1.0, :]
        sum_nodes, sum_edges = ilp_optimisation(test_nodes, test_edges,
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
