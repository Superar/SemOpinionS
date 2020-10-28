import pylcs
import numpy as np
from smatch import get_amr_match, compute_f
from sklearn.cluster import SpectralClustering
from src.document import Document

path = '../Corpora/OpiSums-PT/Textos_AMR/O-Apanhador-no-Campo-de-Centeio/O-Apanhador-no-Campo-de-Centeio.parsed'
doc = Document.read(path)


def pairwise_lcs(X):
    matrix = np.zeros((len(X), len(X)))
    for i, snt1 in enumerate(X):
        matrix[i, list(range(i+1))] = pylcs.lcs2_of_list(snt1, X[:i+1])
    return matrix + np.triu(matrix.T, 1)


def pairwise_smatch(X):
    matrix = np.zeros((len(X), len(X)))
    for i, amr1 in enumerate(X):
        for j, amr2 in enumerate(X[:i+1]):
            match = get_amr_match(amr1, amr2)
            _, _, f = compute_f(*match)
            matrix[i, j] = f
    return matrix + np.triu(matrix.T, 1)


def pairwise_concept_cov(X):
    matrix = np.zeros((len(X), len(X)))
    for i, amr1 in enumerate(X):
        for j, amr2 in enumerate(X[:i+1]):
            match_instances = get_amr_match(amr1, amr2,
                                            justinstance=True)
            match_attribs = get_amr_match(amr1, amr2,
                                          justattribute=True)
            match = tuple(map(sum, zip(match_instances, match_attribs)))
            _, r, _ = compute_f(*match)
            matrix[i, j] = r
    return matrix + np.triu(matrix.T, 1)


amrs = list()
for _, _, amr in doc:
    amrs.append(str(amr))

# snts = list()
# for _, snt, _ in doc:
#     snts.append(snt)

clustering = SpectralClustering(n_clusters=5, affinity='precomputed')
# affinity_matrix = pairwise_smatch(amrs)
# affinity_matrix = pairwise_lcs(snts)
affinity_matrix = pairwise_concept_cov(amrs)
data_label = clustering.fit_predict(affinity_matrix)

selected = list()
for c in np.unique(data_label):
    items = np.where(data_label == c)[0]
    cluster_kernel = affinity_matrix[items][:, items]
    kernel_mean = cluster_kernel.mean(axis=1)
    top_snts = np.argsort(kernel_mean)[-3:]
    top_idx = items[top_snts]

    selected.extend(top_idx)

new_doc = Document([doc.corpus[i] for i in selected])
print(len(doc.corpus))
print(len(new_doc.corpus))
