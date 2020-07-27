import os
import tempfile
import networkx as nx
import numpy as np
from ..document import Document
from ..alignment import Alignment
from ..openie import OpenIE
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from itertools import combinations, product

# Read corpus
corpus = Document.read(
    '../Corpora/OpiSums-PT/Textos_AMR/O-Apanhador-no-Campo-de-Centeio/O-Apanhador-no-Campo-de-Centeio.parsed')
alignment = Alignment.read_giza(
    '../Corpora/AMR-PT-OP/AMR-PT-OP-MANUAL/AMR_Aligned.keep')
open_ie_path = '../Corpora/OpenIEOut'

# Fit IDF counts
tf_idf_corpus_path = '../Corpora/Reviews/corpus'
texts = [os.path.join(tf_idf_corpus_path, fn)
         for fn in os.listdir(tf_idf_corpus_path)]
tf_idf = CountVectorizer(input='filename',
                         tokenizer=lambda txt: word_tokenize(txt, language='portuguese'))
df_counts = tf_idf.fit_transform(texts)
num_docs = df_counts.shape[0]

df_counts[df_counts > 0] = 1
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

# Preprocessing
concept_alignments = dict()
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

    # Get words aligned to each concept
    sent_alignment = alignment.get_alignments(snt.lower())
    for concept_var in amr.get_concept_nodes():
        concept = amr.nodes[concept_var]['label']
        if concept in sent_alignment:
            if concept in concept_alignments:
                concept_alignments[concept] |= set(sent_alignment[concept])
            else:
                concept_alignments[concept] = set(sent_alignment[concept])
merged_graph = corpus.merge_graphs()

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
        concept_scores[concept] = tf * np.log((num_docs/(df + 1)))  # TF-IDF
concept_scores = Counter(concept_scores)

important_concepts = concept_scores.most_common(10)  # Most important concepts
important_concepts = [n for n, _ in important_concepts]

selected_data = list()  # Of tuples (concept_1, concept_2, sentence_id, path)
for c1, c2 in combinations(important_concepts, 2):
    for doc in corpus:
        # Search both concepts in the graph
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
            # doc.amr.draw('selected_paths/{}_{}_{}.pdf'.format(doc.id,c1,c2), highlight_path=selected_path)
            selected_data.append((c1, c2, doc.id, selected_path))
            break

# Get OpenIE tuples
for _, _, sent_id, path in selected_data:
    prod, doc, sent = sent_id.split('.')
    triples_path = os.path.join(open_ie_path, prod, doc) + '.csv'
    triples = OpenIE.read_csv(triples_path).get_triples(int(sent)+1)

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

    expanded_path = list()
    if not selected_triples.empty:
        reverse_alignments = alignment.get_reverse_alignments(corpus[sent_id].snt)
        longest_triple = selected_triples['sent_len'].idxmax()
        longest_triple = selected_triples.loc[longest_triple]
        for w in longest_triple['vocab']:
            if w in reverse_alignments:
                for c in reverse_alignments[w]:
                    expanded_path.append(corpus[sent_id].amr.get_label_node(c))
    else:
        # Retornar o AMR da sentenca
        expanded_path = corpus[sent_id].amr.nodes()
    
    print(path)
    print(expanded_path)
