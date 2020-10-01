import argparse
import os
import re
from importlib import import_module
from collections import Counter
from src.document import Document
from src.alignment import Alignment
from src.openie import OpenIE

parser = argparse.ArgumentParser(
    description='SemOpinionS - Semantic Opinion Summarization'
)

parser.add_argument(
    '--method', '-m',
    help='Summarization method to execute',
    required=True
)

parser.add_argument(
    '--corpus', '-c',
    help='AMR parsed corpus to be summarized',
    required=True
)

parser.add_argument(
    '--alignment', '-a',
    help='Alignment file',
    required=True
)

parser.add_argument(
    '--alignment_format', '-af',
    help='Alignment file format to enable reading',
    required=True,
    choices=['giza', 'jamr']
)

parser.add_argument(
    '--gold', '-g',
    help='Gold summary corpus to evaluate',
    required=False
)

parser.add_argument(
    '--openie', '-oie',
    help='OpenIE triples csv file path',
    required=False
)

parser.add_argument(
    '--tfidf',
    help='File to a large corpus from which to calculate TF-IDF counts',
    required=False
)

parser.add_argument(
    '--training', '-t',
    help='Training set inputs for some methods',
    required=False
)

parser.add_argument(
    '--target', '-tt',
    help='Training set target (gold summaries) for some methods',
    required=False
)

parser.add_argument(
    '--output', '-o',
    help='Output directory',
    required=True
)

args = parser.parse_args()

if args.alignment and not args.alignment_format:
    parser.error(
        'Please provide alignment file format (--alignment_format/-af)')

if not os.path.exists(args.output):
    os.mkdir(args.output)

# Run summarization method
corpus = Document.read(args.corpus)

# Read alignment file
if args.alignment_format == 'giza':
    alignment = Alignment.read_giza(args.alignment)
else:
    alignment = Alignment.read_jamr(args.alignment)

kwargs = dict()
if args.openie:
    open_ie = OpenIE.read_csv(args.openie)
    kwargs['open_ie'] = open_ie
if args.tfidf:
    kwargs['tf_idf_corpus_path'] = args.tfidf
if args.training:
    kwargs['training'] = args.training
if args.target:
    kwargs['target'] = args.target

method = import_module('src.methods.' + args.method)
summary_graph = method.run(corpus, alignment, **kwargs)

# Get alignments for each concept in the corpus
concept_to_words = dict()
for id_, snt, amr in corpus:
    snt_alignment = alignment.get_alignments(snt)
    for c in snt_alignment:
        if c in concept_to_words:
            concept_to_words[c].update(snt_alignment[c])
        else:
            concept_to_words[c] = Counter(snt_alignment[c])

# Save summarization result graph
save_summary_path = os.path.join(args.output, args.method + '.amr')
with open(save_summary_path, 'w', encoding='utf-8') as file_:
    file_.write(str(summary_graph))

# Save summary BOW from alignments
summary_text = list()
for n in summary_graph.get_concept_nodes():
    concept = summary_graph.get_node_label(n)
    if concept in concept_to_words:
        summary_text.append(concept_to_words[concept].most_common(n=1)[0][0])
for c in summary_graph.get_constant_nodes():
    if c.startswith('"'):
        summary_text.append(c.strip('"'))

save_summary_text_path = os.path.join(args.output, args.method + '.bow')
with open(save_summary_text_path, 'w', encoding='utf-8') as file_:
    file_.write(' '.join(summary_text))
    file_.write('\n')

# Write evaluation files
if args.gold:
    for filename in os.listdir(args.gold):
        filepath = os.path.join(args.gold, filename)
        summary_sents = list()
        with open(filepath, encoding='utf-8') as file_:
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

        name, _ = os.path.splitext(filename)
        # Save AMR graph
        save_summary_path = os.path.join(args.output, name + '.amr')
        with open(save_summary_path, 'w', encoding='utf-8') as file_:
            file_.write(str(gold_summary_graph))

        # Save BOW from alignemnts
        summary_text = list()
        for n in gold_summary_graph.get_concept_nodes():
            concept = gold_summary_graph.get_node_label(n)
            if concept in concept_to_words:
                summary_text.append(
                    concept_to_words[concept].most_common(n=1)[0][0])
        for c in gold_summary_graph.get_constant_nodes():
            if c.startswith('"'):
                summary_text.append(c.strip('"'))

        save_summary_text_path = os.path.join(args.output, name + '.bow')
        with open(save_summary_text_path, 'w', encoding='utf-8') as file_:
            file_.write(' '.join(summary_text))
            file_.write('\n')
