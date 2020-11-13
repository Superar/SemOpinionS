import pandas as pd
from pathlib import Path
from joblib import dump, load
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
from itertools import repeat
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from .score_optimization import prepare_training_data, get_tf_idf, integrate_sentiment, get_concept_alignments, calculate_features
from .DohareEtAl2018 import create_final_summary
from ..document import Document
from ..alignment import Alignment
from ..amr import AMR
from ..sentlex import SentimentLexicon
from .DohareEtAl2018 import get_tf_idf as Dohare_tf_idf


def train(training_path: Path, target_path: Path,
          sentlex: SentimentLexicon, tf_idf_path: Path,
          alignment: Alignment) -> DecisionTreeClassifier:
    training_files = list(training_path.iterdir())
    target_files = list(target_path.iterdir())
    tf_idf = get_tf_idf(training_path, target_path, tf_idf_path)

    with ThreadPoolExecutor(max_workers=2 * cpu_count() + 1) as executor:
        result = executor.map(prepare_training_data,
                              training_files,
                              target_files,
                              repeat(sentlex),
                              repeat(alignment),
                              repeat(tf_idf))

    data = pd.concat(result)
    feats = data.loc[:, data.columns != 'class']
    objective = data.loc[:, 'class']

    clf = DecisionTreeClassifier()
    clf.fit(feats, objective)
    return clf


def run(corpus: Document, alignment: Alignment, **kwargs) -> AMR:
    training_path = kwargs.get('training')
    target_path = kwargs.get('target')
    sentlex_path = kwargs.get('sentlex')
    sentlex = SentimentLexicon.read_oplexicon(sentlex_path)
    tf_idf_path = kwargs.get('tfidf')
    model_path = kwargs.get('model')
    open_ie = kwargs.get('open_ie')
    output_path = kwargs.get('output')

    if not model_path and (training_path and target_path):
        model = train(training_path, target_path,
                      sentlex, tf_idf_path,
                      alignment)
        dump(model, output_path / 'model.joblib')
    elif model_path:
        model = load(model_path)

    if corpus:
        merged_graph = corpus.merge_graphs()
        integrate_sentiment(merged_graph, sentlex)
        tf_idf = Dohare_tf_idf(corpus, tf_idf_path)
        concept_alignments = get_concept_alignments(corpus, alignment)
        test_feats = calculate_features(merged_graph, sentlex,
                                        concept_alignments, tf_idf)

        predictions = model.predict(test_feats)
        selected_nodes = test_feats.index[predictions]
        important_concepts = [merged_graph.get_node_label(n)
                              for n in selected_nodes]
        summary_graph = create_final_summary(corpus, important_concepts,
                                             alignment, open_ie)
        return summary_graph
