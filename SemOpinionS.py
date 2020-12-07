import argparse
import re
from importlib import import_module
from collections import Counter
from pathlib import Path
from src.document import Document
from src.alignment import Alignment
from src.openie import OpenIE

# Set arguments
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
    required=False
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
    type=Path,
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
    type=Path,
    required=False
)

parser.add_argument(
    '--training', '-t',
    help='Training set inputs for some methods',
    type=Path,
    required=False
)

parser.add_argument(
    '--target', '-tt',
    help='Training set target (gold summaries) for some methods',
    type=Path,
    required=False
)

parser.add_argument(
    '--model', '-mo',
    help='Pre-trained model (ILP weights or ML model)',
    required=False
)

parser.add_argument(
    '--loss', '-l',
    help='Loss function to be used during ILP weight learning',
    required=False,
    choices=['perceptron', 'ramp'],
    default='perceptron'
)

parser.add_argument(
    '--sentlex', '-s',
    help='Path to a sentiment lexicon',
    type=Path,
    required=False
)

parser.add_argument(
    '--similarity', '-sim',
    help='Similarity metric to use for Spectral Clustering',
    required=False,
    choices=['lcs', 'smatch', 'concept_coverage'],
    default='lcs'
)

parser.add_argument(
    '--output', '-o',
    help='Output directory',
    type=Path,
    required=True
)

args = parser.parse_args()

# Check arguments
if args.alignment and not args.alignment_format:
    parser.error(
        'Please provide alignment file format (--alignment_format/-af)')

if not args.output.exists():
    args.output.mkdir()

# Read corpus file
if args.corpus:
    corpus = Document.read(args.corpus)
else:
    corpus = None

# Read alignment file
if args.alignment_format == 'giza':
    alignment = Alignment.read_giza(args.alignment)
else:
    alignment = Alignment.read_jamr(args.alignment)

# Set extra arguments for different methods
kwargs = dict()
if args.openie:
    open_ie = OpenIE.read_csv(args.openie)
    kwargs['open_ie'] = open_ie
if args.tfidf:
    kwargs['tfidf'] = args.tfidf
if args.training:
    kwargs['training'] = args.training
if args.target:
    kwargs['target'] = args.target
if args.output:
    kwargs['output'] = args.output
if args.model:
    kwargs['model'] = args.model
kwargs['loss'] = args.loss
if args.sentlex:
    kwargs['sentlex'] = args.sentlex
kwargs['similarity'] = args.similarity

# Import the selected method
method = import_module('src.methods.' + args.method)
summary_graph = method.run(corpus, alignment, **kwargs) # Run the method

# Get alignments for each concept in the corpus
concept_to_words = dict()
if corpus:
    for _, snt, _ in corpus:
        snt_alignment = alignment.get_alignments(snt)
        for c in snt_alignment:
            if c in concept_to_words:
                concept_to_words[c].update(snt_alignment[c])
            else:
                concept_to_words[c] = Counter(snt_alignment[c])

# Save summarization result graph
if summary_graph:
    save_summary_path = (args.output / args.method).with_suffix('.amr')
    with save_summary_path.open('w', encoding='utf-8') as file_:
        file_.write(str(summary_graph))

# Save summary BOW from alignments
if concept_to_words and summary_graph:
    summary_text = list()
    for n in summary_graph.get_concept_nodes():
        concept = summary_graph.get_node_label(n)
        if concept in concept_to_words:
            summary_text.append(concept_to_words[concept].most_common(n=1)[0][0])
    for c in summary_graph.get_constant_nodes():
        if c.startswith('"'):
            summary_text.append(c.strip('"'))

    save_summary_text_path = (args.output / args.method).with_suffix('.bow')
    with save_summary_text_path.open('w', encoding='utf-8') as file_:
        file_.write(' '.join(summary_text))
        file_.write('\n')

# Write evaluation files
if args.gold:
    for filepath in args.gold.iterdir():
        summary_sents = list()
        with filepath.open(encoding='utf-8') as file_:
            for sent in file_:
                # Sentence ID between <>s
                info = re.search(r'<([^>]+)>', sent)
                if info is not None:
                    id_ = info.group(1)
                    sent_amr = corpus[id_]
                    if sent_amr is not None:
                        summary_sents.append(sent_amr)
        summary_corpus = Document(summary_sents)
        gold_summary_graph = summary_corpus.merge_graphs()

        name = filepath.stem
        # Save AMR graph
        save_summary_path = (args.output / name).with_suffix('.amr')
        with save_summary_path.open('w', encoding='utf-8') as file_:
            file_.write(str(gold_summary_graph))

        # Save BOW from alignemnts
        if concept_to_words:
            summary_text = list()
            for n in gold_summary_graph.get_concept_nodes():
                concept = gold_summary_graph.get_node_label(n)
                if concept in concept_to_words:
                    summary_text.append(
                        concept_to_words[concept].most_common(n=1)[0][0])
            for c in gold_summary_graph.get_constant_nodes():
                if c.startswith('"'):
                    summary_text.append(c.strip('"'))

            save_summary_text_path = (args.output / name).with_suffix('.bow')
            with save_summary_text_path.open('w', encoding='utf-8') as file_:
                file_.write(' '.join(summary_text))
                file_.write('\n')
