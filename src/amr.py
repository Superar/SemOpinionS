import penman
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations


class AMR(nx.MultiDiGraph):
    def __init__(self):
        super().__init__()

    @classmethod
    def load_penman(cls, penman_g):
        amr = cls()
        for r in penman_g.triples:
            if r[1] == ':instance':
                amr.add_node(r[0], label=r[2])
            else:
                amr.add_edge(r[0], r[2], r[1], label=r[1])
        if penman_g.top:
            amr.add_edge(penman_g.top, penman_g.top, ':TOP', label=':TOP')
        return amr

    def __str__(self):
        return penman.encode(self.as_penman_graph())

    def as_penman_graph(self, keep_top_edges=False):
        defined_variables = set()
        triples = list()
        for s, t, r in self.edges:
            if r != ':TOP':
                triples.append((s, r, t))

                # Add variables included in the edge
                if s in self.variables() and s not in defined_variables:
                    triples.append((s, ':instance', self.nodes[s]['label']))
                    defined_variables.add(s)
                if t in self.variables() and t not in defined_variables:
                    triples.append((t, ':instance', self.nodes[t]['label']))
                    defined_variables.add(t)
            elif keep_top_edges:
                # Only add :TOP edges if flag is up
                triples.append((s, r, t))
        
        # Add remaining variables
        for var in self.variables() - defined_variables:
            triples.append((var, ':instance', self.nodes[var]['label']))
        return penman.Graph(triples=triples, top=self.get_top())

    def as_weighted_DiGraph(self):
        dg = nx.DiGraph(self)

        for e in dg.edges:
            dg.edges[e]['weight'] = self.number_of_edges(e[0], e[1])

        return dg

    def variables(self):
        return {n for n in self.nodes if self.nodes[n]}

    def add_concept(self, concept, variable=None):
        if not variable:
            variable = concept[0]
            var_count = 0
            while variable + str(var_count) in self.variables():
                var_count += 1
            variable += str(var_count)
        elif variable in self.variables():
            raise RuntimeError('Variable already exists')
        self.add_node(variable, label=concept)

        return variable

    def get_concept_nodes(self):
        return [n for n in self.nodes if self.nodes[n]]

    def get_constant_nodes(self):
        return [n for n in self.nodes if not self.nodes[n]]

    def get_top(self):
        for s, _, r in self.edges:
            if r == ':TOP':
                return s

    def get_node_depth(self, node):
        return nx.shortest_path_length(self.to_undirected(),
                                       source=node,
                                       target=self.get_top())

    def get_node_label(self, node):
        try:
            return self.nodes[node]['label']
        except KeyError:
            return node

    def get_label_node(self, label):
        for n in self.nodes:
            try:
                if self.nodes[n]['label'] == label:
                    return n
            except KeyError:
                if n == label:
                    return n

    def copy(self):
        return AMR.load_penman(self.as_penman_graph(keep_top_edges=True))

    def merge(self, amr_graph, collapse_ner=False, collapse_date=False):
        assert isinstance(amr_graph, AMR)

        # NER nodes should not be merged
        self._collapse_ner_nodes()
        amr_graph._collapse_ner_nodes()

        # Date nodes should not be merged
        self._collapse_date_nodes()
        amr_graph._collapse_date_nodes()

        merge_graph = self.copy()
        concepts = {self.nodes[n]['label']
                    for n in self.nodes if self.nodes[n]}

        # Add new nodes
        for n in amr_graph.get_concept_nodes():
            concept = amr_graph.nodes[n]['label']
            if concept not in concepts:
                concepts.add(concept)

                if n in merge_graph.variables():
                    merge_graph.add_concept(concept)
                else:
                    merge_graph.add_concept(concept, variable=n)

        # Add new edges
        for src, tgt, role in amr_graph.edges:
            source = amr_graph.nodes[src]['label']
            source_var = None
            try:
                target = amr_graph.nodes[tgt]['label']
                target_var = None
            except KeyError:
                # Attribute
                target = tgt
                target_var = target

            for n in merge_graph.nodes:
                if merge_graph.nodes[n]:
                    concept = merge_graph.nodes[n]['label']
                    if not source_var and concept == source:
                        source_var = n
                    if not target_var and concept == target:
                        target_var = n
                    if source_var and target_var:
                        break

            merge_graph.add_edge(source_var, target_var, key=role, label=role)

        # Restore NER nodes
        if not collapse_ner:
            self._uncollapse_ner_nodes()
            amr_graph._uncollapse_ner_nodes()
            merge_graph._uncollapse_ner_nodes()

        if not collapse_date:
            self._uncollapse_date_nodes()
            amr_graph._uncollapse_date_nodes()
            merge_graph._uncollapse_date_nodes()

        return merge_graph

    def draw(self, path='amr.pdf', highlight_subgraph_nodes=[], highlight_subgraph_edges=[]):
        if highlight_subgraph_nodes:
            for n in highlight_subgraph_nodes:
                self.nodes[n]['color'] = 'red'
            for e in self.subgraph(highlight_subgraph_nodes).edges:
                self.edges[e]['color'] = 'red'
        
        if highlight_subgraph_edges:
            for s, t in highlight_subgraph_edges:
                self.nodes[s]['color'] = 'red'
                self.nodes[t]['color'] = 'red'
                for e in self.out_edges([s], keys=True):
                    if e[1] == t:
                        self.edges[e]['color'] = 'red'

        pydot_graph = nx.drawing.nx_pydot.to_pydot(self)
        pydot_graph.write_pdf(prog='dot', path=path)

        if highlight_subgraph_nodes or highlight_subgraph_edges:
            for n, d in self.nodes(data=True):
                if 'color' in d:
                    del self.nodes[n]['color']
            for s, t, d in self.edges(data=True):
                if 'color' in d:
                    del self.edges[s, t, d['label']]['color']

    def _collapse_ner_nodes(self):
        ner_nodes = list()
        for n in self.get_concept_nodes():
            for succ in self.successors(n):
                try:
                    if self.nodes[succ]['label'] == 'name':
                        ner_nodes.append([n, succ])
                        break
                except KeyError:
                    # Not a concept node
                    continue

        constants = list()

        for ner, name in ner_nodes:
            subgraph = [ner, name]
            name_ops = list(self.successors(name))
            constants.extend(name_ops)
            subgraph.extend(name_ops)

            collapsed_label = 'NER:'
            collapsed_label += self.nodes[ner]['label']
            collapsed_label += '.' + self.nodes[name]['label'] + '.'
            collapsed_label += '.'.join(name_ops)

            self.remove_node(name)
            self.nodes[ner]['label'] = collapsed_label

        self.remove_nodes_from(constants)

    def _uncollapse_ner_nodes(self):
        for n in self.get_concept_nodes():
            if self.nodes[n]['label'].startswith('NER:'):
                ner_data = self.nodes[n]['label'].lstrip('NER:').split('.')

                self.nodes[n]['label'] = ner_data[0]
                var = self.add_concept(ner_data[1])
                self.add_edge(n, var, ':name', label=':name')

                for i, name in enumerate(ner_data[2:], start=1):
                    if name:
                        self.add_node(name)
                        role = ':op{}'.format(i)
                        self.add_edge(var, name, role, label=role)

    def _collapse_date_nodes(self):
        to_remove = list()

        for n in self.get_concept_nodes():
            if self.nodes[n]['label'] == 'date-entity':
                collapsed_label = 'DATE'

                for suc in self.successors(n):
                    if not self.nodes[suc]:
                        # Only collapse constant nodes
                        for e in self.get_edge_data(n, suc):
                            collapsed_label += e
                            collapsed_label += '.' + suc
                        to_remove.append(suc)
                    else:
                        # Also collapse weekday nodes
                        for e in self.get_edge_data(n, suc):
                            if e == ':weekday':
                                collapsed_label += e
                                collapsed_label += '.' + \
                                    self.nodes[suc]['label']
                                to_remove.append(suc)
                self.nodes[n]['label'] = collapsed_label

        self.remove_nodes_from(to_remove)

    def _uncollapse_date_nodes(self):
        for n in self.get_concept_nodes():
            if self.nodes[n]['label'].startswith('DATE:'):
                data = self.nodes[n]['label'].lstrip('DATE:').split(':')

                self.nodes[n]['label'] = 'date-entity'

                for d in data:
                    rel, val = d.split('.')
                    rel = ':{}'.format(rel)
                    if rel == ':weekday':
                        # Only collapsed weekday relations have concepts
                        var = self.add_concept(val)
                        self.add_edge(n, var, rel, label=rel)
                    else:
                        # All other nodes are constants
                        self.add_node(val)
                        self.add_edge(n, val, rel, label=rel)
