from collections import namedtuple
from pathlib import Path
import penman
import networkx as nx
from .amr import AMR


class Document(object):
    '''
    Class that reads a file with AMR graphs in penman notation.

    Attributes:
        corpus: list of tuples (id, sentence, AMR)
    '''
    doc_item = namedtuple('DocumentSent', ['id', 'snt', 'amr'])

    def __init__(self, corpus, corpus_path=None):
        self.corpus = corpus
        self.path = corpus_path

    def __iter__(self):
        return iter(self.corpus)

    def __getitem__(self, item):
        for doc in self:
            if doc.id == item:
                return doc
    
    def __contains__(self, item):
        return bool(self.__getitem__(item))

    @classmethod
    def read(cls, corpus_path):
        '''
        Creates an object from a file containing ids, sentences and
        AMR graphs in penman notation.

        Parameters:
            corpus_path (str): Path of the file to be read

        Returns:
            Document: An object with all read AMR graphs
        '''
        corpus = list()
        with open(corpus_path, encoding='utf-8') as corpusfile:
            corpusstr = corpusfile.read()
        for penman_g in penman.loads(corpusstr):
            amr = AMR.load_penman(penman_g)
            corpus.append(cls.doc_item(penman_g.metadata['id'],
                                       penman_g.metadata['snt'],
                                       amr))
        return cls(corpus, Path(corpus_path))

    def merge_graphs(self, collapse_ner=False, collapse_date=False):
        '''
        Merges all AMR graphs in the current document into a single representation.

        Parameters:
            collapse_ner (bool, default False): Wheter to keep all NE nodes collapsed
            collapse_date (bool, default False): Wheter to keep all date nodes collapsed

        Return:
            AMR: A single representation of all AMR graphs in the document merged
        '''
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
        nodes_to_remove = [n for n in merge_graph.nodes()
                           if n not in largest_component]
        merge_graph.remove_nodes_from(nodes_to_remove)

        return merge_graph
