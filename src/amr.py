import penman
import networkx as nx


class AMR(nx.MultiDiGraph):
    """
    Class that represents an AMR graph. Inherits from Networkx' MultiDiGraph.
    Should be created using the method `load_penman` from Penman's Graph.

    Attributes:
        penman (penman.Graph): Graph representation in PENMAN.
                               If it is not available, it is set to None
    """

    def __init__(self):
        super().__init__()
        self.penman = None

    @classmethod
    def load_penman(cls, penman_g: penman.Graph):
        """
        Creates an AMR graph based on a networkx.MultiDigraph from a penman.Graph.

        Parameters:
            penman_g (penman.Graph): Graph object containing all triples to be included in the graph

        Returns:
            AMR: AMR graph corresponding to the input penman.
                 The input penman is set to the penman attribute.
        """
        amr = cls()
        amr.set_penman(penman_g)
        for r in penman_g.triples:
            if r[1] == ':instance':
                amr.add_node(r[0], label=r[2])
            else:
                amr.add_edge(r[0], r[2], r[1], label=r[1])
        if penman_g.top:
            amr.add_edge(penman_g.top, penman_g.top, ':TOP', label=':TOP')
        return amr

    def __str__(self):
        """
        Creates a PENMAN string corresponding to the current object.
        If there is a "gold" penamn attribute, it is used.
        Otherwise a dummy penman is constructed using the as_penman_graph method.

        Returns:
            str: PENMAN encoding of the current graph
        """
        if self.penman is not None:
            tree = penman.configure(self.penman)
            # Reset metadata to avoid comments
            tree.metadata = dict()
            return penman.format(tree)
        else:
            return penman.encode(self.as_penman_graph())

    def set_penman(self, penman_g: penman.Graph) -> None:
        """Sets the penman attribute."""
        self.penman = penman_g

    def as_penman_graph(self, keep_top_edges: bool = False) -> penman.Graph:
        """
        Returns the current graph as a penman.Graph object.

        Parameters:
            keep_top_edges (bool, default False): Keep original :TOP relations in the penman.Graph

        Returns:
            penman.Graph: Corresponding penman.Graph.
        """
        if self.penman is not None:
            return self.penman
        else:
            defined_variables = set()
            triples = list()
            for s, t, r in self.edges:
                if r != ':TOP':
                    triples.append((s, r, t))

                    # Add variables included in the edge
                    if s in self.variables() and s not in defined_variables:
                        triples.append(
                            (s, ':instance', self.nodes[s]['label']))
                        defined_variables.add(s)
                    if t in self.variables() and t not in defined_variables:
                        triples.append(
                            (t, ':instance', self.nodes[t]['label']))
                        defined_variables.add(t)
                elif keep_top_edges:
                    # Only add :TOP edges if flag is up
                    triples.append((s, r, t))

            # Add remaining variables
            for var in self.variables() - defined_variables:
                triples.append((var, ':instance', self.nodes[var]['label']))
            return penman.Graph(triples=triples, top=self.get_top())

    def as_weighted_DiGraph(self) -> nx.DiGraph:
        """
        Creates a networkx' DiGraph with edge weights corresponding to the number
        of original edges between each pair of nodes.

        Returns:
            netorkx.DiGraph
        """
        dg = nx.DiGraph(self)

        for e in dg.edges:
            dg.edges[e]['weight'] = self.number_of_edges(e[0], e[1])

        return dg

    def as_levi_graph(self):
        """ Turns all relations into nodes """
        new_graph = self.copy()
        edges_to_remove = list()
        for s, t, r in self.edges:
            if r != ':TOP':
                c_var = new_graph.add_concept(r.lstrip(':'))
                new_graph.add_edge(s, c_var, key='in', label='in')
                new_graph.add_edge(c_var, t, key='out', label='out')
                edges_to_remove.append((s, t, r))
        new_graph.remove_edges_from(edges_to_remove)
        return new_graph

    def variables(self) -> set:
        """Returns a set of node names (variables) if they are not constants"""
        return {n for n in self.nodes if self.nodes[n]}

    def add_concept(self, concept: str, variable: str = None) -> str:
        """
        Adds a new concept (node) in the graph.

        Parameters:
            concept (str): concept to be included
            variable (str, default None): variable under which the concept will be named

        Returns:
            str: The final variable name indicating the new node included in the graph
        """
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

    def get_concept_nodes(self) -> list:
        """Returns a list of node names (variables) if they are not constants"""
        return [n for n in self.nodes if self.nodes[n] and 'label' in self.nodes[n]]

    def get_constant_nodes(self) -> list:
        """Returns a list of node names if they are constants"""
        return [n for n in self.nodes if not self.nodes[n]]

    def get_top(self) -> str:
        """Returns the root node of the graph. This is represented by a self-loop labeled :TOP."""
        for s, _, r in self.edges:
            if r == ':TOP':
                return s

    def get_node_depth(self, node: str) -> int:
        """Returns the depth (length of shortest path to the root) of a given node."""
        return nx.shortest_path_length(self.to_undirected(),
                                       source=node,
                                       target=self.get_top())

    def get_node_label(self, node: str) -> str:
        """
        Return the label if a given node.
        If it is a constant, the node name is also the label.
        """
        try:
            return self.nodes[node]['label']
        except KeyError:
            return node

    def get_label_node(self, label: str) -> str:
        """Returns the node which contains the given label."""
        for n in self.nodes:
            try:
                if self.nodes[n]['label'] == label:
                    return n
            except KeyError:
                if n == label:
                    return n

    def copy(self):
        """
        Returns a new AMR object containing the same nodes and edges.
        The penman attribute is, however, set to None.
        """
        copied_amr = AMR.load_penman(self.as_penman_graph(keep_top_edges=True))
        copied_amr.set_penman(None)
        return copied_amr

    def merge(self, amr_graph, collapse_ner: bool = False, collapse_date: bool = False):
        """
        Merges two AMR graphs together according to their node labels.
        NE and date nodes are collapsed before proceeding with the merging.
        Thus, they are only merged if their attributes are the same.

        Parameters:
            amr_graph (AMR): Another graph to be merged
            collapse_ner (bool, default False): Wheter to keep all NE nodes collapsed
            collapse_date (bool, default False): Wheter to keep all date nodes collapsed

        Returns:
            AMR: A new AMR graph containing all nodes and edges from both input AMRs
        """
        assert isinstance(amr_graph, AMR)

        # NER nodes should not be merged
        self.collapse_ner_nodes()
        amr_graph.collapse_ner_nodes()

        # Date nodes should not be merged
        self.collapse_date_nodes()
        amr_graph.collapse_date_nodes()

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
            self.uncollapse_ner_nodes()
            amr_graph.uncollapse_ner_nodes()
            merge_graph.uncollapse_ner_nodes()

        # Restore date nodes
        if not collapse_date:
            self.uncollapse_date_nodes()
            amr_graph.uncollapse_date_nodes()
            merge_graph.uncollapse_date_nodes()

        return merge_graph

    def draw(self, path: str = 'amr.pdf',
             highlight_subgraph_nodes: list = None,
             highlight_subgraph_edges: list = None) -> None:
        """
        Draw the current AMR graph to a PDF file using DOT.
        The graph can also have some subgraph highlighted in red.

        Parameters:
            path (str, default amr.pdf): Path to which draw the file
            highlight_subgraph_nodes (list): List of nodes to highlight with all edges between them
            highlight_subgraph_edges (list): List of edges to highlight with all their nodes
        """
        if highlight_subgraph_nodes is None:
            highlight_subgraph_nodes = list()
        if highlight_subgraph_edges is None:
            highlight_subgraph_edges = list()

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

        # Restore original colours
        if highlight_subgraph_nodes or highlight_subgraph_edges:
            for n, d in self.nodes(data=True):
                if 'color' in d:
                    del self.nodes[n]['color']
            for s, t, d in self.edges(data=True):
                if 'color' in d:
                    del self.edges[s, t, d['label']]['color']

    def collapse_ner_nodes(self) -> None:
        """
        Collapses all sets of NE subgraphs into a sigle node for each NE name.

        For example, the following NE:
        (c / city
            :wiki "New_York_City"
            :name (n / name
                :op1 "New"
                :op2 "York"))

        Will become:
        (N1 / NER:city.name."New"."York"
            :wiki "New_York_City")
        """
        ner_nodes = list()
        for n in self.get_concept_nodes():
            for succ in self.successors(n):
                try:
                    # Node has a 'name' node as child
                    if self.nodes[succ]['label'] == 'name':
                        # Linked by a ':name' relation
                        if ':name' in self.get_edge_data(n, succ):
                            ner_nodes.append([n, succ])
                            break
                except KeyError:
                    # Not a concept node
                    continue

        constants = list()

        for ner, name in ner_nodes:
            subgraph = [ner, name]
            # Assuming all successors are leaf and constant nodes
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

    def uncollapse_ner_nodes(self) -> None:
        """Restores all collapsed NE nodes into multiple node representations"""
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

    def collapse_date_nodes(self) -> None:
        """
        Collapses all sets of date-tentity subgraphs into a sigle node for each date.

        For example, the following NE:
        (d / date-entity
            :year 2012
            :month 2
            :day 29)

        Will become:
        (D1 / DATE:year.2:month.2:day.29)
        """
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

    def uncollapse_date_nodes(self) -> None:
        """Restores all collapsed date-entity nodes into multiple node representations"""
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
