from .amr import AMR
from collections import namedtuple
import penman
import networkx as nx


class Document(object):
    """Class that reads a file with AMR graphs in penman notation"""
    doc_item = namedtuple('DocumentSent', ['id', 'snt', 'amr'])

    def __init__(self, corpus):
        self.corpus = corpus

    def __iter__(self):
        return iter(self.corpus)

    def __getitem__(self, item):
        for doc in self:
            if doc.id == item:
                return doc

    @classmethod
    def read(cls, corpus_path):
        corpus = list()
        with open(corpus_path, encoding='utf-8') as corpusfile:
            corpusstr = corpusfile.read()
        for penman_g in penman.loads(corpusstr):
            amr = AMR.load_penman(penman_g)
            corpus.append(cls.doc_item(penman_g.metadata['id'],
                                       penman_g.metadata['snt'],
                                       amr))
        return cls(corpus)

    def merge_graphs(self, collapse_ner=False, collapse_date=False):
        merge_graph = AMR()
        for amr in self.corpus:
            merge_graph = merge_graph.merge(amr.amr,
                                            collapse_ner=collapse_ner,
                                            collapse_date=collapse_date)

        # Place multi-sentence root
        new_root = merge_graph.add_concept('multi-sentence')

        top = merge_graph.get_top()
        root_number = 1
        while top:
            merge_graph.remove_edge(top, top, ':TOP')
            merge_graph.add_edge(new_root, top,
                                 ':snt{}'.format(root_number),
                                 label=':snt{}'.format(root_number))
            top = merge_graph.get_top()
            root_number += 1
        merge_graph.add_edge(new_root, new_root, ':TOP', label=':TOP')

        # Remove disconnected nodes
        # This should not affect well-formed AMR graphs
        largest_component = max(nx.connected_components(merge_graph.to_undirected()),
                                key=len)
        nodes_to_remove = [
            n for n in merge_graph.nodes() if n not in largest_component]
        merge_graph.remove_nodes_from(nodes_to_remove)

        return merge_graph
