import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from joblib import dump, load
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from itertools import repeat
from typing import Union
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from .score_optimization import (prepare_training_data, get_tf_idf, integrate_sentiment,
                                 get_concept_alignments, calculate_features, get_aspect_list)
from .DohareEtAl2018 import create_final_summary
from ..document import Document
from ..alignment import Alignment
from ..amr import AMR
from ..sentlex import SentimentLexicon
from .DohareEtAl2018 import get_tf_idf as Dohare_tf_idf


def train(training_path: Path, target_path: Path,
          sentlex: SentimentLexicon, tf_idf_path: Path,
          alignment: Alignment, type_: str, levi: bool = False,
          aspects: Path = None) -> Union[DecisionTreeClassifier, RandomForestClassifier, SVC, MLPClassifier]:
    """
    Train the Machine Learning model to select imporant nodes (concepts)
    from the given data. The model is selected through the `type_` parameter.

    Paramters:
        training_path (Path): The corpus to use as training.
        target_path (Path): The corresponding corpus to use as target.
        sentlex (SentimentLexicon): Sentiment lexicon mapping concepts to their sentiment polarity.
        tf_idf_path (Path): Path to a larger corpus from which to calculate Document Frequency.
        alignment (Alignment): The concept alignments for both train and target corpora.
        type_ (str): Which ML technique to use (decision_tree, random_forest, svm, mlp).
        levi (bool): Whether or not to use Levi Graphs
                     (convert edges to nodes, so that the ML model can select them too).
        aspects (Path): Include aspect annotation as a feature. Ignored if None.

    Returns:
        DecisionTreeClassifier, RandomForestClassifier, SVC, MLPClassifier: Trained model.
    """
    training_files = list(training_path.iterdir())
    target_files = list(target_path.iterdir())
    tf_idf = get_tf_idf(training_path, target_path, tf_idf_path)

    with ThreadPoolExecutor(max_workers=2 * cpu_count() + 1) as executor:
        result = executor.map(prepare_training_data,
                              training_files,
                              target_files,
                              repeat(sentlex),
                              repeat(alignment),
                              repeat(tf_idf),
                              repeat(levi),
                              repeat(aspects))

    data = pd.concat(result).fillna(0)
    feats = data.loc[:, data.columns != 'class']
    objective = data.loc[:, 'class']

    if type_ == 'decision_tree':
        clf = DecisionTreeClassifier()
    elif type_ == 'random_forest':
        clf = RandomForestClassifier()
    elif type_ == 'svm':
        clf = SVC()
    elif type_ == 'mlp':
        clf = MLPClassifier()
    clf.fit(feats, objective)
    return clf


def run(corpus: Document, alignment: Alignment, **kwargs) -> AMR:
    """
    Run method.

    Parameters:
        corpus(Document): The corpus upon which the summarization process will be applied.
        alignment(Alignment): Concept alignments corresponding to the `corpus`.

    Returns:
        AMR: Summary graph created from the `corpus`.
    """
    training_path = kwargs.get('training')
    target_path = kwargs.get('target')
    sentlex_path = kwargs.get('sentlex')
    sentlex = SentimentLexicon.read_oplexicon(sentlex_path)
    tf_idf_path = kwargs.get('tfidf')
    model_path = kwargs.get('model')
    open_ie = kwargs.get('open_ie')
    machine_learning = kwargs.get('machine_learning')
    levi = kwargs.get('levi')
    aspects = kwargs.get('aspects')
    output_path = kwargs.get('output')

    # Train or load model
    if not model_path and (training_path and target_path):
        model = train(training_path, target_path,
                      sentlex, tf_idf_path,
                      alignment, machine_learning, levi,
                      aspects)
        dump(model, output_path / 'model.joblib')
    elif model_path:
        model = load(model_path)

    # Test
    if corpus:
        merged_graph = corpus.merge_graphs()
        if levi:
            merged_graph = merged_graph.as_levi_graph()
        integrate_sentiment(merged_graph, sentlex)
        tf_idf = Dohare_tf_idf(corpus, tf_idf_path)
        concept_alignments = get_concept_alignments(corpus, alignment)

        aspect_list = get_aspect_list(aspects, corpus.path.name)
        test_feats = calculate_features(merged_graph, sentlex,
                                        concept_alignments, tf_idf,
                                        aspect_list)

        predictions = model.predict(test_feats)
        selected_nodes = test_feats.index[predictions].to_list()

        if levi:
            changed = True
            selected_relations = [t for _, t, d
                                  in merged_graph.in_edges(selected_nodes, data=True)
                                  if d['label'] == 'in']
            while changed:
                # Include nodes associated with selected relations in a greedy manner
                changed = False
                for rel in selected_relations:
                    in_concept = list(merged_graph.predecessors(rel))[0]
                    out_concept = list(merged_graph.successors(rel))[0]
                    if in_concept in selected_nodes and out_concept not in selected_nodes:
                        # Include out_concept because in_concept was selected
                        selected_nodes.append(out_concept)
                        changed = True
                    elif out_concept in selected_nodes and in_concept not in selected_nodes:
                        # Include in_concept because out_concept was selected
                        selected_nodes.append(in_concept)
                        changed = True

            # Get only those relations with both concepts also selected
            sum_relations = [r for r in selected_relations
                             if all([c in selected_nodes
                                     for c in merged_graph.to_undirected().neighbors(r)])]
            sum_triples = [(list(merged_graph.predecessors(r))[0],
                            list(merged_graph.successors(r))[0],
                            f':{merged_graph.nodes[r]["label"]}') for r in sum_relations]
            summary_graph = corpus.merge_graphs().edge_subgraph(sum_triples).copy()

            # Remove disconnected nodes
            largest_component = max(nx.connected_components(summary_graph.to_undirected()),
                                    key=len)
            disconnected_nodes = list()
            for node in summary_graph.nodes():
                if node not in largest_component:
                    disconnected_nodes.append(node)
            summary_graph.remove_nodes_from(disconnected_nodes)
            return summary_graph
        else:
            important_concepts = [merged_graph.get_node_label(n)
                              for n in selected_nodes]
            summary_graph = create_final_summary(corpus, important_concepts,
                                                alignment, open_ie)
            return summary_graph
