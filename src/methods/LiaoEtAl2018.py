import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from multiprocessing import cpu_count
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import SpectralClustering
from nltk import word_tokenize
from pylcs import lcs
from smatch import get_amr_match, compute_f
from random import sample
from typing import Tuple, Dict
from scipy.sparse import csr_matrix
from pathlib import Path
import pandas as pd
import numpy as np
from .LiuEtAl2015 import (update_weights,
                          calculate_node_data,
                          calculate_edge_data,
                          graph_local_representations,
                          ilp_optimisation)
from .DohareEtAl2018 import preprocess, score_concepts
from ..document import Document
from ..alignment import Alignment
from ..amr import AMR
from .DohareEtAl2018 import get_tf_idf as Dohare_tf_idf
from .LiuEtAl2015 import prepare_training_data as Liu_prepare_data


def include_tf_idf(local_repr, scores, concept_alignments, tf_idf):
    """
    Given a matrix of attributes for nodes/edges, incorporate TF-IDF counts as a new attribute.
    The attribute is binirized in 4 points: -4.5, 0, 3, 7.5; as these were, respectively,
    the minimum (using a maximum of 1.5 interquartile range), 1st quartile, 2nd quartile and 3rd quartile
    values in our training set.
    This function also adds another feature indicating if the concept is in the larger corpus used to
    calculate the DF values.

    Parameters:
        local_repr (DataFrame): Local representations (attribute matrix) for each node/edge.
        scores (Counter): TF-IDF values for each concept.
        concept_alignments (dict): Mapping from concepts to words.
        tf_idf (CountVectorizer): CountVectorizer used to calculate the TF-IDF values.
                                  Used to get vocabulary mappings into indices.
    
    Returns:
        DataFrame: `local_repr` extended with the TF-IDF values for each row.
    """
    for idx, feats in local_repr.iterrows():
        concept = feats['concept']

        if concept in scores:
            score = scores[feats['concept']]
        else:
            score = -float('inf')  # Will set all threasholds to 0

        # Binarize the value (4 additional features)
        for t in [-4.5, 0, 3, 7.5]:
            if score >= t:
                local_repr.loc[[idx], f'tf_idf_{t}'] = 1.0
            else:
                local_repr.loc[[idx], f'tf_idf_{t}'] = 0.0

        # Indicate if the concept (any word to which it is aligned) is present in the larger vocabulary.
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


def prepare_training_data(training_path: Path, gold_path: Path,
                          alignment: Alignment, tf_idf_counts: tuple) -> pd.DataFrame:
    """
    Create the training instances, one for each node/edge in each graph in both training_path and gold_path.
    Both training and gold paths must be aligned, so that the `gold_path` documents corresponds to
    the target value (the summary) of the `training_path` document.

    Parameters:
        training_path (Path): Training document.
        gold_path (Path): Target/summary document.
        alignment (Alignment): Concept alignments containing information of both train and target documents.
        tf_idf_counts (tuple): A tuple returned by the LiaoEtAl2018.get_tf_idf() function.
    
    Returns:
        DataFrame: Matrix containing the attributes representations for each node/edge in both training and gold documents.
    """
    # Get TF and DF counts for this specific pair
    tf_idf, tf_counts, df_counts, num_docs, doc_to_index = tf_idf_counts
    doc_tf = tf_counts[doc_to_index[('training', training_path.name)], :]

    # Preprocessing: merge graphs and get alignments
    merged_graph, concept_alignments = preprocess(Document.read(training_path),
                                                  alignment)
    # Calculate TF-IDF for each concept in the merged graph
    scores = score_concepts(merged_graph,
                            (tf_idf, doc_tf, df_counts, num_docs),
                            concept_alignments)

    # Get attributes: the ones from LiuEtAl2015 + TF-IDF
    features = Liu_prepare_data(training_path, gold_path, alignment)
    features = include_tf_idf(features, scores, concept_alignments, tf_idf)

    return features


def train(training_path: Path, gold_path: Path, alignment: Alignment, tf_idf_counts: tuple) -> np.array:
    """
    Train the weights for the scoring method using ILP and AdaGrad. The preprocessing is done parallelly.
    Each node/edge from each AMR graph is represented as a set of binary attributes.
    The importance score of a node/edge is given by the linear combination of its attributes given a weight vector.
    The weight vector is initialized as a vector of 1, then it is updated via AdaGrad
    using a ramp loss function through supervised learning.

    Parameters:
        training_path (Path): The corpus to use as training.
        gold_path (Path): The corpus to use as target.
        alignment (Alignment): The concept alignments for both train and target corpora.
        tf_idf_counts (tuple): A tuple returned by the LiaoEtAl2018.get_tf_idf() function.
    
    Returns:
        array: Optimized weights for the scoring of nodes and edges.
    """
    # Create training instances parallelly through the prepare_training_data function
    with ThreadPoolExecutor(max_workers=cpu_count() - 1) as executor:
        # Organize arguments for mapping
        train_filepaths = list()
        target_filepaths = list()
        for instance_path in training_path.iterdir():
            train_filepaths.append(instance_path)
            target_filepaths.append(gold_path / instance_path.name)
        alignment_arg = repeat(alignment)
        tf_idf_arg = repeat(tf_idf_counts)

        # Create training and target representations
        result = executor.map(prepare_training_data,
                              train_filepaths,
                              target_filepaths,
                              alignment_arg,
                              tf_idf_arg)

    # Combine all results from the parallel processing
    # Also provide one-hot encoding for concept attributes
    local_reprs_df = pd.get_dummies(pd.concat(result),
                                    columns=['concept',
                                             'node1_concept',
                                             'node2_concept'],
                                    dtype=np.float32)
    # Get pairs of train-target instances
    pairs_groups = local_reprs_df.groupby(by='name')

    # Initialize weights
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


def get_tf_idf(training_path: Path, gold_path: Path, tf_idf_path: Path) -> Tuple[CountVectorizer, csr_matrix, csr_matrix, int, Dict[str, int]]:
    """
    Calculate both Term Frequency (TF) and Document Frequency (DF) counts for the words in given corpora. 
    `training_path` and `gold_path are the ones to calculate TF, while `tf_idf_path`
    is the one to use for DF counts.
    This also returns the number of documents used to calculate the DF and a mapping of each
    document in `training_path` and `gold_path` to their corresponding index in the TF matrix.

    Parameters:
        training_path (Path): The training corpus to calculate the TF.
        gold_path (Path): The target/gold corpus to calculate the TF.
        tf_idf_path (Path): The corpus to calculate the DF.
    
    Returns:
        tuple(CountVectorizer, csr_matrix, csr_matrix, int, dict): Tuple containing the CountVectorizer
        (sklearn) object used in the counting, the matrices for both TF and DF (in this order),
        the number of documents processed to calculate the DF and, finally, the mapping for each document
        in training and gold corpora into indices of the TF matrix.
    """
    # Count DF
    texts = list(tf_idf_path.iterdir())
    tf_idf = CountVectorizer(input='filename',
                             tokenizer=lambda txt: word_tokenize(txt, language='portuguese'))
    df_counts = tf_idf.fit_transform(texts)
    num_docs = df_counts.shape[0]

    df_counts[df_counts > 0] = 1
    df_counts = np.sum(df_counts, axis=0)

    training_paths = list(training_path.iterdir())
    gold_paths = list(gold_path.iterdir())

    # Write temporary files for both training and gold texts
    tf_paths = list()
    for file_ in training_paths + gold_paths:
        doc = Document.read(file_)
        tmp, tmp_name = tempfile.mkstemp()
        with open(tmp_name, 'w', encoding='utf-8') as tmp_file:
            for _, snt, _ in doc:
                tmp_file.write(snt)
                tmp_file.write('\n')
        tf_paths.append((tmp, tmp_name))

    # Count TF
    tf_counts = tf_idf.transform([p for _, p in tf_paths])
    for tmp, tmp_name in tf_paths:
        os.close(tmp)
        os.remove(tmp_name)

    # Create a mapping from training and gold files into a matrix index
    doc_to_index = {('training', f.name): i
                    for i, f in enumerate(training_path.iterdir())}
    doc_to_index.update({('gold', f.name): len(training_paths) + i
                         for i, f in enumerate(gold_path.iterdir())})
    return tf_idf, tf_counts, df_counts, num_docs, doc_to_index


def calculate_similarity_matrix(corpus: Document, metric: str) -> pd.DataFrame:
    """
    Calculates a similarity matrix between each pair of sentences in the `corpus`.
    The similarity can be determined by three different types of metrics:
        - Longest Common Subsequence (lcs): Number of words in common (in the same place) between two sentences.
        - Smatch (smatch): Smatch score between the AMR graphs of both sentences.
        - Concept Coverage (concept_coverage): The number of concepts in common between the AMAR graphs of both sentences.
    
    Parameters:
        corpus (Document): Corpus from which to create the similarity matrix.
        metric (str): What kind of metric to calculate the similarity.
    
    Returns:
        DataFrame: Similarity matrix between each pair of sentences.
    """
    ids = [id_ for id_, _, _ in corpus]

    similarity = pd.DataFrame(0, columns=ids, index=ids)
    for id1 in ids:
        for id2 in ids:
            if metric == 'lcs':
                sim = lcs(corpus[id1].snt, corpus[id2].snt)
            elif metric == 'smatch':
                match = get_amr_match(str(corpus[id1].amr),
                                      str(corpus[id2].amr))
                _, _, sim = compute_f(*match)
            elif metric == 'concept_coverage':
                amr1 = str(corpus[id1].amr)
                amr2 = str(corpus[id2].amr)
                match_instances = get_amr_match(amr1, amr2,
                                                justinstance=True)
                match_attribs = get_amr_match(amr1, amr2,
                                              justattribute=True)
                match = tuple(map(sum, zip(match_instances, match_attribs)))
                _, sim, _ = compute_f(*match)
            similarity.loc[id1, id2] = sim
    return similarity


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
    tf_idf_corpus_path = kwargs.get('tfidf')
    similarity = kwargs.get('similarity')

    # Check arguments
    if not weights_path and not (training_path and gold_path):
        raise ValueError('LiaoEtAl2018 method requires either training and '
                         'target arguments or pre-trained weights')
    if not tf_idf_corpus_path:
        raise ValueError('LiaoEtAl2018 method requires '
                         'a larger corpus to calculate TF-IDF')

    # Train or load weights
    if not weights_path and (training_path and gold_path):
        counts = get_tf_idf(training_path, gold_path, tf_idf_corpus_path)
        weights = train(training_path, gold_path, alignment, counts)
        weights.to_csv(output_path / 'weights.csv')
    elif weights_path:
        weights = pd.read_csv(weights_path, index_col=0, squeeze=True)

    if corpus:
        # Clustering
        similarity_matrix = calculate_similarity_matrix(corpus, similarity)
        clt = SpectralClustering(n_clusters=5, affinity='precomputed')
        clusters = pd.Series(clt.fit_predict(similarity_matrix),
                             index=similarity_matrix.index,
                             name='cluster')
        selected_sentences = list()
        for _, g in clusters.groupby(clusters):
            if len(g.index) <= 5:
                selected_sentences.extend(g.index.to_list())
            else:
                selected_sentences.extend(sample(g.index.to_list(), 5))
        clustered_corpus = Document([corpus[id_]
                                     for id_ in selected_sentences])
        merged_test_graph = clustered_corpus.merge_graphs(collapse_ner=True,
                                                          collapse_date=True)
        test_node_data = calculate_node_data(clustered_corpus, alignment)
        test_edge_data = calculate_edge_data(clustered_corpus)
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
