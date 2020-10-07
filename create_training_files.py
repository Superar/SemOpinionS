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

training_path = corpus_path.parent / 'training'
if not training_path.exists():
    training_path.mkdir()

gold_path = Path(args.gold)
target_path = corpus_path.parent / 'target'
if not target_path.exists():
    target_path.mkdir()

for i in range(len(list(gold_path.iterdir()))):
    training_filepath = training_path / f'{name}_{i + 1}.txt'
    with training_filepath.open('w', encoding='utf-8') as file_:
        for id_, snt, amr in corpus:
            file_.write(f'# ::id {id_}\n')
            file_.write(f'# ::snt {snt}\n')
            file_.write(f'{amr}\n\n')

for i, d in enumerate(gold_path.iterdir()):
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

    target_filepath = target_path / f'{name}_{i + 1}.txt'
    with target_filepath.open('w', encoding='utf-8') as file_:
        for id_, snt, amr in summary_corpus:
            file_.write(f'# ::id {id_}\n')
            file_.write(f'# ::snt {snt}\n')
            file_.write(f'{amr}\n\n')
