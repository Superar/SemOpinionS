from .DohareEtAl2018 import get_tf_idf, preprocess, get_important_paths, expand_paths, get_summary_graph
from collections import Counter
from ..amr import AMR
from ..document import Document
from ..alignment import Alignment


def score_concepts(graph: AMR, counts: tuple, concept_alignments: dict) -> Counter:
    """
    Calculate TF counts for each node (concept) in `graph` according to their aligned words.

    Parameters:
        graph (AMR): Graph which contains the concept to be scored.
        counts (tuple): A tuple returned by the DohareEtAl2018.get_tf_idf() function.
        concept_alignments (dict): A dictionary that maps concepts into a list of words.
    
    Returns:
        Counter: All TF counts for each concept. If the concept does not exist, the count is 0.
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


def run(corpus: Document, alignment: Alignment, **kwargs: dict) -> AMR:
    """
    Run method.

    Parameters:
        corpus (Document): The corpus upon which the summarization process will be applied.
        alignment (Alignment): Concept alignments corresponding to the `corpus`.
    
    Returns:
        AMR: Summary graph created from the `corpus`.
    """
    open_ie = kwargs.get('open_ie')
    tf_idf_corpus_path = kwargs.get('tfidf')

    tf_idf = get_tf_idf(corpus, tf_idf_corpus_path)
    merged_graph, concept_alignments = preprocess(corpus, alignment)

    concept_scores = score_concepts(merged_graph, tf_idf, concept_alignments)
    concepts = concept_scores.most_common(10)  # Most important concepts
    important_concepts = [n for n, _ in concepts]

    selected_paths = get_important_paths(corpus, important_concepts)
    expanded_paths = expand_paths(corpus, alignment, selected_paths, open_ie)
    summary_graph = get_summary_graph(corpus, expanded_paths)
    return summary_graph
