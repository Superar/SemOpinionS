import os
import tempfile
import networkx as nx
import numpy as np
from ..document import Document
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

# Read corpus
corpus = Document.read(
    '..\Corpora\OpiSums-PT\Textos_AMR\Iphone-5\Iphone-5.parsed')
# corpus = Document.read(
#     '..\Corpora\OpiSums-PT\Textos_AMR\O-Apanhador-no-Campo-de-Centeio\O-Apanhador-no-Campo-de-Centeio.parsed')

concept_counts = Counter()

for _, _, amr in corpus:
    for c in amr.get_concept_nodes():
        concept = amr.nodes[c]['label']
        if concept in concept_counts:
            concept_counts[concept] += 1
        else:
            concept_counts[concept] = 1

# First
summary_first = corpus[0]

# First Co-occurrence
n1, n2 = concept_counts.most_common(2)
for id_, snt, amr in corpus:
    concepts = {amr.nodes[c]['label'] for c in amr.get_concept_nodes()}
    if n1[0] in concepts and n2[0] in concepts:
        summary_first_cooccurrence = (id_, snt, amr)
        break
