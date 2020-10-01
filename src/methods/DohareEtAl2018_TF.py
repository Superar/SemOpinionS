from .DohareEtAl2018 import get_tf_idf, preprocess, get_important_paths, expand_paths, get_summary_graph
from collections import Counter
from amr import AMR


def score_concepts(graph: AMR, counts: tuple, concept_alignments: dict) -> Counter:
    """
    Calculate TF counts for each node (concept) in `graph` according to their aligned words.

    `counts` is a tuple returned by the get_tf_idf() function.
    `concept_alignments` is a dictionary that maps concepts into a list of words.
    """
    tf_idf, tf_counts, _, _ = counts
    # Get score for each node
    scores = Counter()
    for c in graph.get_concept_nodes():
        concept = graph.nodes[c]['label']
        if concept in concept_alignments:
            for w in concept_alignments[concept]:
                if w in tf_idf.vocabulary_:
                    scores[concept] += tf_counts[0, tf_idf.vocabulary_[w]]
    return scores


def run(corpus, alignment, **kwargs):
    open_ie = kwargs.get('open_ie')
    tf_idf_corpus_path = kwargs.get('tf_idf_corpus_path')

    tf_idf = get_tf_idf(corpus, tf_idf_corpus_path)
    merged_graph, concept_alignments = preprocess(corpus, alignment)

    concept_scores = score_concepts(merged_graph, tf_idf, concept_alignments)
    concepts = concept_scores.most_common(10)  # Most important concepts
    important_concepts = [n for n, _ in concepts]

    selected_paths = get_important_paths(corpus, important_concepts)
    expanded_paths = expand_paths(corpus, alignment, selected_paths, open_ie)
    summary_graph = get_summary_graph(corpus, expanded_paths)
    return summary_graph
