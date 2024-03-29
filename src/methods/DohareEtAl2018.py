import os
import tempfile
import networkx as nx
import numpy as np
from ..document import Document
from ..alignment import Alignment
from ..amr import AMR
from ..openie import OpenIE
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from collections import Counter
from itertools import combinations, product
from typing import Tuple, Dict
from pathlib import Path


def get_tf_idf(corpus: Document, tf_idf_corpus_path: Path) -> Tuple[CountVectorizer, csr_matrix, csr_matrix, int]:
    """
    Calculate both Term Frequency (TF) and Document Frequency (DF) counts for the words in given corpora.
    `corpus` is the one to calculate TF, while `tf_idf_corpus_path` is the one to use for DF counts.
    This also returns the number of documents used to calculate the DF.

    Parameters:
        corpus (Document): The corpus to calculate the TF.
        tf_idf_corpus_path (Path): The corpus to calculate the DF.
    
    Returns:
        tuple(CountVectorizer, csr_matrix, csr_matrix, int): Tuple containing the CountVectorizer
        (sklearn) object used in the counting, the matrices for both TF and DF (in this order) and,
        finally the number of documents processed to calculate the DF.
    """
    # Fit IDF counts
    texts = list(tf_idf_corpus_path.iterdir())
    tf_idf = CountVectorizer(input='filename',
                             tokenizer=lambda txt: word_tokenize(txt, language='portuguese'))
    df_counts = tf_idf.fit_transform(texts)
    num_docs = df_counts.shape[0]

    df_counts[df_counts > 0] = 1  # Indicate only presence
    # number of docs in which token is present
    df_counts = np.sum(df_counts, axis=0)

    # Write file to calculate TF counts
    tmp, tmp_name = tempfile.mkstemp()
    with open(tmp_name, 'w', encoding='utf-8') as tmp_file:
        for _, snt, _ in corpus:
            tmp_file.write(snt)
            tmp_file.write('\n')
    tf_counts = tf_idf.transform([tmp_name])
    os.close(tmp)
    os.remove(tmp_name)

    return tf_idf, tf_counts, df_counts, num_docs


def get_concept_alignments(corpus: Document, alignment: Alignment) -> Dict[str, str]:
    """
    Creates a mapping from concepts to words according to the `alignment` for the given `corpus`.

    Parameters:
        corpus (Document): The corpus beign summarized.
        alignment (Alignment): Concept alignments for the `corpus`.
    
    Returns:
        dict(str, str): Dictionary mapping concepts (node labels) to words in the sentences from `corpus`.
    """
    concept_alignments = dict()
    for _, snt, amr in corpus:
        # Get words aligned to each concept
        sent_alignment = alignment.get_alignments(snt.lower())
        if sent_alignment:
            for concept_var in amr.get_concept_nodes():
                concept = amr.nodes[concept_var]['label']
                if concept in sent_alignment:
                    if concept in concept_alignments:
                        concept_alignments[concept] |= set(
                            sent_alignment[concept])
                    else:
                        concept_alignments[concept] = set(
                            sent_alignment[concept])

    return concept_alignments


def preprocess(corpus: Document, alignment: Alignment) -> Tuple[AMR, Dict[str, str]]:
    """
    Preprocessing and data extraction. Removes cycles from the AMR graphs in the `corpus`.
    Also removes possible disconnected nodes. Finally it merges all sentence graphs into one.
    This function also creates a mapping from concepts to their aligned words.

    Parameters:
        corpus(Document): The corpus beign summarized.
        alignment(Alignment): Concept alignments for the `corpus`.

    Returns:
        tuple(AMR, dict): Tuple containing the merged graph and the mapping of concepts to words.
    """
    # Preprocessing
    for id_, snt, amr in corpus:
        # Remove cycles
        for cycle in nx.simple_cycles(amr.as_weighted_DiGraph()):
            for node in cycle:
                if node != amr.get_top():
                    amr.remove_node(node)

        # Remove disconnected nodes
        largest_component = max(nx.connected_components(amr.to_undirected()),
                                key=len)
        for node in amr.nodes():
            if node not in largest_component:
                amr.remove_node(node)
    concept_alignments = get_concept_alignments(corpus, alignment)
    merged_graph = corpus.merge_graphs()
    return merged_graph, concept_alignments


def score_concepts(merged_graph: AMR, counts: tuple, concept_alignments: Dict[str, str]) -> Counter:
    """
    Calculate TF-IDF counts for each node(concept) in `merged_graph` according to their aligned words.

    Parameters:
        merged_graph(AMR): Graph which contains the concept to be scored.
        counts(tuple): A tuple returned by the DohareEtAl2018.get_tf_idf() function.
        concept_alignments(dict): A dictionary that maps concepts into a list of words.

    Returns:
        Counter: All TF-IDF scores for each concept. If the concept does not exist, the score is 0.
    """
    tf_idf, tf_counts, df_counts, num_docs = counts
    # Get score for each node
    concept_scores = dict()
    for c in merged_graph.get_concept_nodes():
        concept = merged_graph.nodes[c]['label']
        if concept in concept_alignments:
            tf = 0
            df = 0
            for w in concept_alignments[concept]:
                try:
                    tf += tf_counts[0, tf_idf.vocabulary_[w]]
                    df += df_counts[0, tf_idf.vocabulary_[w]]
                except KeyError:
                    pass
            concept_scores[concept] = tf * \
                np.log((num_docs/(df + 1)))  # TF-IDF
    concept_scores = Counter(concept_scores)

    return concept_scores


def get_important_paths(corpus: Document, important_concepts: list) -> list:
    """
    Get the best path between each pair of concepts in `important_concepts`.
    The best path is defined as the one closest to the root in the first sentence that
    contains both concepts.

    Parameters:
        corpus(Document): The corpus being summarized.
        imporant_concepts(list): List of concepts(actual labels, not variables) selected
        to be included in the summary.

    Returns:
        list: List of tuples(concept_1, concept_2, sentence_id, path)
        containing the selected paths.
    """
    selected_data = list()  # Of tuples (concept_1, concept_2, sentence_id, path)
    for c1, c2 in combinations(important_concepts, 2):
        for doc in corpus:
            # Search first sentence with both concepts in the graph
            nodes_c1 = list()
            nodes_c2 = list()
            for c in doc.amr.get_concept_nodes():
                if doc.amr.nodes[c]['label'] == c1:
                    nodes_c1.append(c)
                    continue
                if doc.amr.nodes[c]['label'] == c2:
                    nodes_c2.append(c)

            if nodes_c1 and nodes_c2:
                # Search path between both concepts closest to the root
                # Use all combinations of nodes with the given concepts
                # Ignoring the direction of edges
                selected_path = list()
                selected_path_depth = float('inf')
                for n1, n2 in product(nodes_c1, nodes_c2):
                    paths = nx.all_simple_paths(doc.amr.to_undirected(),
                                                source=n1,
                                                target=n2)
                    for p in paths:
                        p_depth = min([doc.amr.get_node_depth(c) for c in p])
                        if p_depth < selected_path_depth:
                            selected_path = p
                            selected_path_depth = p_depth
                selected_data.append((c1, c2, doc.id, selected_path))
                break
    return selected_data


def expand_paths(corpus: Document, alignment: Alignment,
                 selected_data: list, open_ie: OpenIE) -> list:
    """
    Expand selected paths using  OpenIE tuples.

    Parameters:
        corpus(Document): The corpus being summarized.
        alignment(Alignment): Concept alignments for the `corpus`.
        selected_data(list): List of tuples(concept_1, concept2, sentence_id, path)
        containing the path between two concepts in a sentence from `corpus`.
        open_ie(OpenIE): OpenIE triples information for the `corpus`.

    Returns:
        list: List of tuples(sentence_id, expanded_path) with each path from `selected_data` expanded.
    """
    expanded_data = list()  # Of tuples (sent_id, expanded_path)
    for _, _, sent_id, path in selected_data:
        triples = open_ie.get_triples(sent_id)

        path_concepts = [corpus[sent_id].amr.get_node_label(c) for c in path]
        snt_alignments = alignment.get_alignments(corpus[sent_id].snt)

        path_aligned_words = list()
        for c in path_concepts:
            if c in snt_alignments:
                path_aligned_words.extend(snt_alignments[c])
        path_aligned_words = set(path_aligned_words)

        def get_intersection(w): return w.intersection(path_aligned_words)
        triples_intersection = triples['vocab'].apply(get_intersection)

        selected_triples = triples.loc[triples_intersection ==
                                       path_aligned_words, :]

        expanded_path = set()
        if not selected_triples.empty:
            reverse_alignments = alignment.get_reverse_alignments(
                corpus[sent_id].snt)
            longest_triple = selected_triples['sent_len'].idxmax()
            longest_triple = selected_triples.loc[longest_triple]
            for w in longest_triple['vocab']:
                if w in reverse_alignments:
                    for c in reverse_alignments[w]:
                        if isinstance(c, tuple):
                            # Aligned role, include both extremes
                            expanded_path.add(c[0])
                            expanded_path.add(c[1])
                        else:
                            # Aligned concept
                            c_node = corpus[sent_id].amr.get_label_node(c)
                            if c_node is not None:
                                expanded_path.add(c_node)
        else:
            # Return the whole sentence AMR
            expanded_path = set(corpus[sent_id].amr.nodes())

        # Expand selected subgraph to ensure connectivity
        undirected = nx.Graph(corpus[sent_id].amr)
        weighted_graph = nx.Graph()
        weighted_graph.add_nodes_from(expanded_path)
        weighted_graph.add_edges_from(
            undirected.subgraph(expanded_path).edges, weight=1)

        for n1, n2 in combinations(expanded_path, 2):
            if not nx.has_path(weighted_graph, n1, n2):
                shortest_path = nx.shortest_path(undirected, n1, n2)
                weighted_graph.add_edge(n1, n2, weight=len(
                    shortest_path), path=shortest_path)

        mst = nx.minimum_spanning_tree(weighted_graph)
        for _, _, data in mst.edges(data=True):
            if 'path' in data:
                expanded_path |= set(data['path'])

        if (sent_id, expanded_path) not in expanded_data:
            expanded_data.append((sent_id, expanded_path))
    return expanded_data


def get_summary_graph(corpus: Document, expanded_data: list) -> AMR:
    """
    Merges all important subgraphs from `corpus` with the information from `expanded_data`.

    Parameters:
        corpus(Document): The corpus being summarized.
        expanded_data(list): List with all important expanded paths.

    Returns:
        AMR: Summary graph.
    """
    # Construct summary graph by merging all subgraphs
    summary_tuples = list()
    for sent_id, path in expanded_data:
        doc = Document.doc_item(sent_id,
                                corpus[sent_id].snt,
                                corpus[sent_id].amr.subgraph(path).copy())
        summary_tuples.append(doc)
    summary_doc = Document(summary_tuples, None)
    summary_graph = summary_doc.merge_graphs()
    return summary_graph


def create_final_summary(corpus: Document, important_concepts: list,
                         alignment: Alignment, open_ie: OpenIE) -> AMR:
    """
    Creates the final summary graph by finding the paths to be included
    between important concepts, as well as expanding these paths using
    OpenIE triples information.

    Parameters:
        corpus(Document): The corpus being summarized.
        important_concepts(list): List with all important nodes to be included in the summary.
        alignment(Alignment): Concept alignments for the `corpus`.
        open_ie(OpenIE): OpenIE triples information.

    Returns:
        AMR: Summary graph.
    """
    selected_paths = get_important_paths(corpus, important_concepts)
    expanded_paths = expand_paths(corpus, alignment, selected_paths, open_ie)
    summary_graph = get_summary_graph(corpus, expanded_paths)
    return summary_graph


def run(corpus: Document, alignment: Alignment, **kwargs: dict) -> AMR:
    """
    Run method.

    Parameters:
        corpus(Document): The corpus upon which the summarization process will be applied.
        alignment(Alignment): Concept alignments corresponding to the `corpus`.

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

    summary_graph = create_final_summary(corpus, important_concepts,
                                         alignment, open_ie)
    return summary_graph
