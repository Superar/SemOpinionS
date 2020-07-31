import argparse
from importlib import import_module
from src.document import Document

parser = argparse.ArgumentParser(
    description='SemOpinionS - Semantic Opinion Summarization'
)

parser.add_argument(
    '--method', '-m',
    help='Summarization method to execute',
    required=True
)

args = parser.parse_args()

corpus_path = '../Corpora/OpiSums-PT/Textos_AMR/O-Apanhador-no-Campo-de-Centeio/O-Apanhador-no-Campo-de-Centeio.parsed'
alignment_path = '../Corpora/AMR-PT-OP/AMR-PT-OP-MANUAL/AMR_Aligned.keep'

method = import_module('src.methods.' + args.method)
summary_graph = method.run(corpus_path, alignment_path)
print(summary_graph)

# if args.method == 'DohareEtAl2017':
#     from literature import DohareEtal2017

# r = Document.read('D:\Documentos\Mestrado\Pesquisa\Corpora\OpiSums-PT\Textos_AMR\Iphone-5\Iphone-5.parsed')

# merge_graph = r.merge_graphs()
# merge_graph.draw()
# merge_digraph = merge_graph.as_weighted_DiGraph()

# score = sorted(nx.pagerank(merge_digraph).items(), key=itemgetter(1), reverse=True)
# nodes = [n for n, _ in score[:10]]
# summary_graph = merge_graph.subgraph(nodes)
