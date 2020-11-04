import argparse
import penman
from sklearn.metrics import classification_report
from amr import AMR

parser = argparse.ArgumentParser()
parser.add_argument('--test', '-t')
parser.add_argument('--gold', '-g')
args = parser.parse_args()

test_g = AMR.load_penman(penman.load(args.test)[0])
gold_g = AMR.load_penman(penman.load(args.gold)[0])

test_nodes = [test_g.get_node_label(n) for n in test_g.nodes]
gold_nodes = [gold_g.get_node_label(n) for n in gold_g.nodes]

nodes_tp = list()
nodes_fp = list()
for n in test_nodes:
    if n in gold_nodes:
        nodes_tp.append(n)
    else:
        nodes_fp.append(n)
nodes_fn = [n for n in gold_nodes if n not in test_nodes]

print('Nodes precision: '
      f'{len(nodes_tp)}/{len(nodes_tp)+len(nodes_fp)} = '
      f'{len(nodes_tp)/(len(nodes_tp)+len(nodes_fp))}')
print('Nodes recall:'
      f'{len(nodes_tp)}/{len(nodes_tp)+len(nodes_fn)} = '
      f'{len(nodes_tp)/(len(nodes_tp)+len(nodes_fn))}')
print('Nodes F-score: '
      f'{len(nodes_tp)/(len(nodes_tp)+0.5*(len(nodes_fp) + len(nodes_fn)))}')

test_edges = [(test_g.get_node_label(s), test_g.get_node_label(t), r)
              for s, t, r in test_g.edges]
gold_edges = [(gold_g.get_node_label(s), gold_g.get_node_label(t), r)
              for s, t, r in gold_g.edges]

edges_tp = list()
edges_fp = list()
for n in test_edges:
    if n in gold_edges:
        edges_tp.append(n)
    else:
        edges_fp.append(n)
edges_fn = [n for n in gold_edges if n not in test_edges]

print('Edges precision: '
      f'{len(edges_tp)}/{len(edges_tp)+len(edges_fp)} = '
      f'{len(edges_tp)/(len(edges_tp)+len(edges_fp))}')
print('Edges recall:'
      f'{len(edges_tp)}/{len(edges_tp)+len(edges_fn)} = '
      f'{len(edges_tp)/(len(edges_tp)+len(edges_fn))}')
print('Edges F-score: '
      f'{len(edges_tp)/(len(edges_tp)+0.5*(len(edges_fp) + len(edges_fn)))}')
