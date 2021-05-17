import re
from pathlib import Path
from src.document import Document

products = ['Galaxy-SIII', 'Iphone-5', 'O-Outro-Lado-da-Meia-Noite', 'O-Apanhador-no-Campo-de-Centeio']

comments_basepath = Path('../Corpora/OpiSums-PT/Textos_AMR/')
summaries_basepath = Path('../Corpora/OpiSums-PT/Sumarios')

for product in products:
    comment_path = comments_basepath / product / (product + '.parsed')
    corpus = Document.read(comment_path)
    amr = corpus.merge_graphs()
    amr.draw(path='{}.pdf'.format(product))

    summaries_path = summaries_basepath / product / 'Extrativos'
    for summary in summaries_path.iterdir():
        summary_sents = list()
        with summary.open(encoding='utf-8') as file_:
            for sent in file_:
                # Sentence ID between <>s
                info = re.search(r'<([^>]+)>', sent)
                if info is not None:
                    id_ = info.group(1)
                    sent_amr = corpus[id_]
                    if sent_amr is not None:
                        summary_sents.append(sent_amr)
        summary_corpus = Document(summary_sents)
        summary_amr = summary_corpus.merge_graphs()

        summary_triples = [(summary_amr.get_node_label(s), summary_amr.get_node_label(t))
                for s, t in summary_amr.edges()]
        comment_triples = [(amr.get_label_node(s), amr.get_label_node(t))
                for s, t in summary_triples]
        amr.draw(path='{}_{}.pdf'.format(product, summary.stem),
                 highlight_subgraph_edges=comment_triples)
