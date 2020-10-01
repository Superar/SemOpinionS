from src.document import Document
from pathlib import Path
import argparse
import networkx as nx
import re

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', '-c')
parser.add_argument('--gold', '-g')

args = parser.parse_args()

corpus_path = Path(args.corpus)
corpus = Document.read(args.corpus)
name = corpus_path.stem

gold_path = Path(args.gold)
target_path = corpus_path.parent / f'{name}.target'
target_file = target_path.open('w', encoding='utf-8')

training_path = corpus_path.parent / f'{name}.training'
with training_path.open('w', encoding='utf-8') as file_:
    for i in range(len(list(gold_path.iterdir()))):
        file_.write(f'# ::id {name}.{i + 1}\n')
        file_.write('# ::snt -\n')
        file_.write(f'{corpus.merge_graphs()}\n\n')

for d in gold_path.iterdir():
    summary_sents = list()
    with d.open('r', encoding='utf-8') as file_:
        for sent in file_:
            # Sentence ID between <>
            info = re.search(r'<([^>]+)>', sent)
            if info is not None:
                id_ = info.group(1)
                sent_amr = corpus[id_]
                if sent_amr is not None:
                    summary_sents.append(sent_amr)
    summary_corpus = Document(summary_sents)
    gold_summary_graph = summary_corpus.merge_graphs()

    target_file.write(f'# ::id {name}.{d.stem}\n')
    target_file.write('# ::snt -\n')
    target_file.write(f'{gold_summary_graph}\n\n')
target_file.close()
