import tempfile
import subprocess
import os
import networkx as nx
from ..document import Document

corpus = Document.read(
    'D:\Documentos\Mestrado\Pesquisa\Corpora\OpiSums-PT\Textos_AMR\O-Apanhador-no-Campo-de-Centeio\O-Apanhador-no-Campo-de-Centeio.parsed')
graph = corpus.merge_graphs().as_weighted_DiGraph()
graph = nx.convert_node_labels_to_integers(graph)

_, temp_in = tempfile.mkstemp()
out_file = 'tmp' + next(tempfile._get_candidate_names())

nx.readwrite.write_weighted_edgelist(graph, temp_in)
subprocess.run(['infomap',
                '--ftree', '--clu', '-d',
                '--out-name', out_file,
                temp_in, os.curdir])
