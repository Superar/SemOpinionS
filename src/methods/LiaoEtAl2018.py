import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from multiprocessing import cpu_count
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize
import pandas as pd
import numpy as np
from .LiuEtAl2015 import update_weights, calculate_node_data, calculate_edge_data, graph_local_representations, ilp_optimisation
from .DohareEtAl2018 import preprocess, score_concepts
from ..document import Document
from .DohareEtAl2018 import get_tf_idf as Dohare_tf_idf
from .LiuEtAl2015 import prepare_training_data as Liu_prepare_data


def include_tf_idf(local_repr, scores, concept_alignments, tf_idf):
    for idx, feats in local_repr.iterrows():
        concept = feats['concept']

        if concept in scores:
            score = scores[feats['concept']]
        else:
            score = -float('inf')  # Will set all threasholds to 0
        for t in [-4.5, 0, 3, 7.5]:
            if score >= t:
                local_repr.loc[[idx], f'tf_idf_{t}'] = 1.0
            else:
                local_repr.loc[[idx], f'tf_idf_{t}'] = 0.0

        in_vocab = False
        if concept in concept_alignments:
            for w in concept_alignments[concept]:
                if w in tf_idf.vocabulary_:
                    in_vocab = True
                    break
        if in_vocab:
            local_repr.loc[[idx], 'larger_corpus'] = 1.0
        else:
            local_repr.loc[[idx], 'larger_corpus'] = 0.0
    return local_repr


def prepare_training_data(training_path, gold_path, alignment, tf_idf_counts):
    tf_idf, tf_counts, df_counts, num_docs, doc_to_index = tf_idf_counts
    doc_tf = tf_counts[doc_to_index[('training', training_path.name)], :]
    merged_graph, concept_alignments = preprocess(Document.read(training_path),
                                                  alignment)
    scores = score_concepts(merged_graph,
                            (tf_idf, doc_tf, df_counts, num_docs),
                            concept_alignments)

    features = Liu_prepare_data(training_path, gold_path, alignment)
    features = include_tf_idf(features, scores, concept_alignments, tf_idf)

    return features


def train(training_path, gold_path, alignment, tf_idf_counts):
    with ThreadPoolExecutor(max_workers=cpu_count() - 1) as executor:
        train_filepaths = list()
        target_filepaths = list()
        for instance_path in training_path.iterdir():
            train_filepaths.append(instance_path)
            target_filepaths.append(gold_path / instance_path.name)
        alignment_arg = repeat(alignment)
        tf_idf_arg = repeat(tf_idf_counts)

        result = executor.map(prepare_training_data,
                              train_filepaths,
                              target_filepaths,
                              alignment_arg,
                              tf_idf_arg)

    local_reprs_df = pd.get_dummies(pd.concat(result),
                                    columns=['concept',
                                             'node1_concept',
                                             'node2_concept'],
                                    dtype=np.float32)
    pairs_groups = local_reprs_df.groupby(by='name')

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
        weights, gold_e, ilp_e = update_weights(weights,
                                                train,
                                                target,
                                                top_train,
                                                loss='ramp')
    return weights


def get_tf_idf(training_path, gold_path, tf_idf_path):
    texts = list(tf_idf_path.iterdir())
    tf_idf = CountVectorizer(input='filename',
                             tokenizer=lambda txt: word_tokenize(txt, language='portuguese'))
    df_counts = tf_idf.fit_transform(texts)
    num_docs = df_counts.shape[0]

    df_counts[df_counts > 0] = 1
    df_counts = np.sum(df_counts, axis=0)

    training_paths = list(training_path.iterdir())
    gold_paths = list(gold_path.iterdir())

    tf_paths = list()
    for file_ in training_paths + gold_paths:
        doc = Document.read(file_)
        tmp, tmp_name = tempfile.mkstemp()
        with open(tmp_name, 'w', encoding='utf-8') as tmp_file:
            for _, snt, _ in doc:
                tmp_file.write(snt)
                tmp_file.write('\n')
        tf_paths.append((tmp, tmp_name))

    tf_counts = tf_idf.transform([p for _, p in tf_paths])
    for tmp, tmp_name in tf_paths:
        os.close(tmp)
        os.remove(tmp_name)

    doc_to_index = {('training', f.name): i
                    for i, f in enumerate(training_path.iterdir())}
    doc_to_index.update({('gold', f.name): len(training_paths) + i
                         for i, f in enumerate(gold_path.iterdir())})
    return tf_idf, tf_counts, df_counts, num_docs, doc_to_index


def run(corpus, alignment, **kwargs):
    training_path = kwargs.get('training')
    gold_path = kwargs.get('target')
    output_path = kwargs.get('output')
    weights_path = kwargs.get('model')
    tf_idf_corpus_path = kwargs.get('tfidf')

    if not weights_path and not (training_path and gold_path):
        raise ValueError('LiaoEtAl2018 method requires either training and '
                         'target arguments or pre-trained weights')
    if not tf_idf_corpus_path:
        raise ValueError('LiaoEtAl2018 method requires'
                         'a larger corpus to calculate TF-IDF')

    if not weights_path and (training_path and gold_path):
        counts = get_tf_idf(training_path, gold_path, tf_idf_corpus_path)
        weights = train(training_path, gold_path, alignment, counts)
        weights.to_csv(output_path / 'weights.csv')
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

        # TF-IDF
        tf_idf = Dohare_tf_idf(corpus, tf_idf_corpus_path)
        preprocessed_graph, concept_alignments = preprocess(corpus, alignment)
        concept_scores = score_concepts(preprocessed_graph,
                                        tf_idf, concept_alignments)
        test_repr = include_tf_idf(test_repr,
                                   concept_scores,
                                   concept_alignments,
                                   tf_idf[0])

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
