import re
import pandas as pd
import networkx as nx
import numpy as np
from pathlib import Path
from multiprocessing import cpu_count
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from amr import AMR
from ..alignment import Alignment
from ..document import Document
from ..sentlex import SentimentLexicon
from .DohareEtAl2018 import get_concept_alignments, score_concepts, create_final_summary
from .LiaoEtAl2018 import get_tf_idf, Dohare_tf_idf
from .DohareEtAl2018_TF import score_concepts as score_concepts_tf, get_important_paths, expand_paths, get_summary_graph


def integrate_sentiment(graph: AMR, sentlex: SentimentLexicon) -> None:
    for n in graph:
        label = graph.get_node_label(n)
        sent = 0
        word = re.sub(r'-d+$', '', label)
        if word in sentlex:
            sent = sentlex[word]
        graph.nodes[n]['sent'] = sent


def calculate_features(graph: AMR,
                       sentlex: SentimentLexicon,
                       concept_alignments: dict,
                       tf_idf_counts: tuple) -> pd.DataFrame:
    nodes = [n for n in graph]
    df = pd.DataFrame(index=nodes)

    # Sentiment
    df['sentiment'] = [graph.nodes[n]['sent'] for n in nodes]
    df['sentiment'] = df['sentiment'].replace(
        {-1: 'negative', 0: 'neutral', 1: 'positive'})
    df = pd.get_dummies(df, columns=['sentiment'])
    # Pagerank
    pagerank = nx.algorithms.pagerank(graph.as_weighted_DiGraph())
    df['pagerank'] = [pagerank[n] for n in nodes]
    # Hits
    hubs, authorities = nx.algorithms.hits(graph)
    df['hubs'] = [hubs[n] for n in nodes]
    df['authorities'] = [authorities[n] for n in nodes]
    # Degree
    degree = graph.degree()
    df['degree'] = [degree[n] for n in nodes]
    avg_neighbour_degree = nx.algorithms.average_neighbor_degree(
        graph)
    df['average neighbour degree'] = [avg_neighbour_degree[n]
                                      for n in nodes]
    # Centrality
    degree_centrality = nx.algorithms.degree_centrality(
        graph)
    df['degree centrality'] = [degree_centrality[n] for n in nodes]
    eigen_centrality = nx.algorithms.eigenvector_centrality(
        graph.as_weighted_DiGraph())
    df['eigenvector centrality'] = [eigen_centrality[n] for n in nodes]
    closeness = nx.algorithms.closeness_centrality(graph)
    df['closeness'] = [closeness[n] for n in nodes]
    betweeness = nx.algorithms.betweenness_centrality(
        graph.as_weighted_DiGraph())
    df['betweeness'] = [betweeness[n] for n in nodes]
    # Clustering
    clustering = nx.algorithms.clustering(graph.as_weighted_DiGraph())
    df['clustering'] = [clustering[n] for n in nodes]
    # TF-IDF
    scores = score_concepts(graph, tf_idf_counts, concept_alignments)
    df['tfidf'] = [scores[graph.get_node_label(n)] for n in nodes]
    # TF
    scores = score_concepts_tf(graph, tf_idf_counts, concept_alignments)
    df['tf'] = [scores[graph.get_node_label(n)] for n in nodes]
    # Node depth
    df['depth'] = [graph.get_node_depth(n) for n in nodes]

    return df


def prepare_training_data(training_path: Path, gold_path: Path,
                          sentlex: SentimentLexicon, alignment: Alignment,
                          tf_idf_counts: tuple) -> pd.DataFrame:
    training_corpus = Document.read(training_path)
    training_graph = training_corpus.merge_graphs()
    integrate_sentiment(training_graph, sentlex)

    gold_corpus = Document.read(gold_path)
    gold_graph = gold_corpus.merge_graphs()
    gold_concepts = [gold_graph.get_node_label(n) for n in gold_graph]

    concept_alignments = get_concept_alignments(training_corpus, alignment)
    tf_idf, tf_counts, df_counts, num_docs, doc_to_index = tf_idf_counts
    doc_tf = tf_counts[doc_to_index[('training', training_path.name)], :]

    # Get features
    df = calculate_features(training_graph, sentlex, concept_alignments,
                            (tf_idf, doc_tf, df_counts, num_docs))
    # Class
    df['class'] = [True if training_graph.get_node_label(n) in gold_concepts
                   else False
                   for n in df.index]

    # Add a new index level to indicate which instance this is
    df = pd.concat({str(training_path.stem): df}, names=['instance'])
    return df


def fitness_function(weights: np.ndarray, data: pd.DataFrame) -> float:
    feats = data.loc[:, data.columns != 'class']
    objective = data.loc[:, 'class']

    combination = (weights * feats).sum(axis='columns')

    selected_nodes = combination.groupby(level=0).nlargest(10).droplevel(0)
    selected_classes = objective.loc[selected_nodes.index]

    selected_counts = selected_classes.groupby(level=0).value_counts()
    if True in selected_counts.index.get_level_values(1):
        true_positives = selected_counts[:, True]
    else:
        true_positives = 0
    if False in selected_counts.index.get_level_values(1):
        false_positives = selected_counts[:, False]
    else:
        false_positives = 0
    relevant_items = objective.groupby(level=0).value_counts()[:, True]

    recall = true_positives / relevant_items
    return recall.fillna(0).mean()


def simulated_annealing_optimization(data: pd.DataFrame,
                                     starting_weights: np.ndarray,
                                     n_iter: int = 1000,
                                     n_neighbors: int = 5,
                                     temp_decay: float = 0.9,
                                     temp_decay_iter: int = 10) -> np.ndarray:
    cur_weights = starting_weights
    cur_fitness = fitness_function(cur_weights, data)
    cur_temp = 1.0

    for i in range(n_iter):
        if (i + 1) % temp_decay_iter == 0:
            cur_temp *= temp_decay

        neighbors = np.tile(cur_weights, (n_neighbors, 1)) + \
            np.random.uniform(-1.0, 1.0, (n_neighbors, cur_weights.shape[0]))
        neighbors_fitness = np.apply_along_axis(fitness_function, 1,
                                                neighbors, data=data)
        best_neighbor = np.argmax(neighbors_fitness)
        if neighbors_fitness[best_neighbor] > cur_fitness:
            # Better solution
            cur_weights = neighbors[best_neighbor, :]
            cur_fitness = neighbors_fitness[best_neighbor]
        else:
            # Update weights anyway according to temperature
            fitness_delta = neighbors_fitness[best_neighbor] - cur_fitness
            probability = np.exp(fitness_delta/cur_temp)
            if probability > np.random.uniform():
                cur_weights = neighbors[best_neighbor, :]
                cur_fitness = neighbors_fitness[best_neighbor]

    return cur_weights


def train(training_path: Path, target_path: Path,
          sentlex: SentimentLexicon, tf_idf_path: Path,
          alignment: Alignment) -> pd.Series:
    training_files = list(training_path.iterdir())
    target_files = list(target_path.iterdir())
    tf_idf = get_tf_idf(training_path, target_path, tf_idf_path)

    with ThreadPoolExecutor(max_workers=2 * cpu_count() + 1) as executor:
        result = executor.map(prepare_training_data,
                              training_files,
                              target_files,
                              repeat(sentlex),
                              repeat(alignment),
                              repeat(tf_idf))

    data = pd.concat(result)

    feats = data.loc[:, data.columns != 'class']
    objective = data.loc[:, 'class']
    num_feats = feats.shape[1]

    weights = np.random.uniform(-1.0, 1.0, num_feats)
    weights = simulated_annealing_optimization(data, weights)
    return pd.Series(weights, index=feats.columns)


def run(corpus: Document, alignment: Alignment, **kwargs) -> AMR:
    training_path = kwargs.get('training')
    target_path = kwargs.get('target')
    sentlex_path = kwargs.get('sentlex')
    sentlex = SentimentLexicon.read_oplexicon(sentlex_path)
    tf_idf_path = kwargs.get('tfidf')
    weights_path = kwargs.get('model')
    open_ie = kwargs.get('open_ie')
    output_path = kwargs.get('output')

    if not weights_path and (training_path and target_path):
        weights = train(training_path, target_path,
                        sentlex, tf_idf_path,
                        alignment)
        weights.to_csv(output_path / 'weights.csv')
    elif weights_path:
        weights = pd.read_csv(weights_path, index_col=0, squeeze=True)

    if corpus:
        merged_graph = corpus.merge_graphs()
        integrate_sentiment(merged_graph, sentlex)
        tf_idf = Dohare_tf_idf(corpus, tf_idf_path)
        concept_alignments = get_concept_alignments(corpus, alignment)
        test_feats = calculate_features(merged_graph, sentlex,
                                        concept_alignments, tf_idf)
        combination = (weights * test_feats).sum(axis='columns')
        selected_nodes = combination.nlargest(10).index
        important_concepts = [merged_graph.get_node_label(n)
                              for n in selected_nodes]

        summary_graph = create_final_summary(corpus, important_concepts,
                                             alignment, open_ie)
        return summary_graph