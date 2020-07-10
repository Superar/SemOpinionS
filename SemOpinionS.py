import argparse
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

__import__('src.methods.' + args.method)

# if args.method == 'DohareEtAl2017':
#     from literature import DohareEtal2017

# r = Document.read('D:\Documentos\Mestrado\Pesquisa\Corpora\OpiSums-PT\Textos_AMR\Iphone-5\Iphone-5.parsed')

# merge_graph = r.merge_graphs()
# merge_graph.draw()
# merge_digraph = merge_graph.as_weighted_DiGraph()

# score = sorted(nx.pagerank(merge_digraph).items(), key=itemgetter(1), reverse=True)
# nodes = [n for n, _ in score[:10]]
# summary_graph = merge_graph.subgraph(nodes)
