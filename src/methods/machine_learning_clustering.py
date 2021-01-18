from sklearn.cluster import SpectralClustering
from joblib import dump, load
from random import sample
import pandas as pd
from .machine_learning import (train, integrate_sentiment, Dohare_tf_idf,
                               get_concept_alignments, calculate_features,
                               create_final_summary, get_aspect_list)
from .LiaoEtAl2018 import calculate_similarity_matrix
from ..document import Document
from ..alignment import Alignment
from ..amr import AMR
from ..sentlex import SentimentLexicon


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
    target_path = kwargs.get('target')
    sentlex_path = kwargs.get('sentlex')
    sentlex = SentimentLexicon.read_oplexicon(sentlex_path)
    tf_idf_path = kwargs.get('tfidf')
    model_path = kwargs.get('model')
    open_ie = kwargs.get('open_ie')
    similarity = kwargs.get('similarity')
    machine_learning = kwargs.get('machine_learning')
    aspects = kwargs.get('aspects')
    output_path = kwargs.get('output')

    # Train of load model
    if not model_path and (training_path and target_path):
        model = train(training_path, target_path,
                      sentlex, tf_idf_path,
                      alignment, machine_learning)
        dump(model, output_path / 'model.joblib')
    elif model_path:
        model = load(model_path)

    # Test
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
        # Run test
        integrate_sentiment(merged_test_graph, sentlex)
        tf_idf = Dohare_tf_idf(corpus, tf_idf_path)
        concept_alignments = get_concept_alignments(corpus, alignment)
        aspect_list = get_aspect_list(aspects, corpus.path.name)
        test_feats = calculate_features(merged_test_graph, sentlex,
                                        concept_alignments, tf_idf,
                                        aspect_list)

        predictions = model.predict(test_feats)
        selected_nodes = test_feats.index[predictions]
        important_concepts = [merged_test_graph.get_node_label(n)
                              for n in selected_nodes]
        summary_graph = create_final_summary(corpus, important_concepts,
                                             alignment, open_ie)
        return summary_graph
