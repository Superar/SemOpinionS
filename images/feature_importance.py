from pathlib import Path
from multiprocessing import cpu_count
from itertools import repeat
from concurrent.futures import ThreadPoolExecutor
from sklearn.inspection import permutation_importance
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import joblib
from src.sentlex import SentimentLexicon
from src.alignment import Alignment
from src.methods.score_optimization import (prepare_training_data, get_tf_idf)

def get_features(training_path: Path, target_path: Path,
                 sentlex: SentimentLexicon, tf_idf_path: Path,
                 alignment: Alignment, levi: bool = False,
                 aspects: Path = None) -> pd.DataFrame:
    training_files = list(training_path.iterdir())
    target_files = list(target_path.iterdir())
    tf_idf = get_tf_idf(training_path, target_path, tf_idf_path)

    with ThreadPoolExecutor(max_workers=2 * cpu_count() + 1) as executor:
        result = executor.map(prepare_training_data,
                              training_files,
                              target_files,
                              repeat(sentlex),
                              repeat(alignment),
                              repeat(tf_idf),
                              repeat(levi),
                              repeat(aspects))

    data = pd.concat(result).fillna(0)
    return data

training_path = Path('../Corpora/Training-AMR/all/training')
target_path = Path('../Corpora/Training-AMR/all/target')
sentlex = SentimentLexicon.read_oplexicon(Path('../Corpora/oplexicon_v3.0/lexico_v3.0.txt'))
tf_idf_path = Path('../Corpora/Reviews/b2w-reviews01_ReLi')
alignment = Alignment.read_jamr(Path('../Corpora/AMR-PT-OP/SPAN-MANUAL/combined_manual_training_target_jamr.txt'))
levi = False
# levi = True
aspects = Path('../Corpora/OpiSums-PT/Aspectos/aspects/all.json')

data = get_features(training_path, target_path,
                    sentlex, tf_idf_path,
                    alignment, levi,
                    aspects)
feats = data.loc[:, data.columns != 'class']
objective = data.loc[:, 'class']

model_dt = joblib.load('../Resultados/out_machine_learning_decision_tree_aspects/model.joblib')
model_rf = joblib.load('../Resultados/out_machine_learning_random_forest_aspects/model.joblib')
model_svm = joblib.load('../Resultados/out_machine_learning_svm_aspects/model.joblib')
model_mlp = joblib.load('../Resultados/out_machine_learning_mlp_aspects/model.joblib')

models = {'Árvore de Decisão': model_dt,
          'Random Forest': model_rf,
          'SVM': model_svm,
          'MLP': model_mlp}

importances = {'Modelo': list(), 'Atributo': list(), 'Execução': list(), 'Importância': list()}
for model in models:
    model_importances = permutation_importance(models[model],
                                               feats, objective,
                                               scoring='recall',
                                               n_repeats=50,
                                               n_jobs=3)
    for i, attribute in enumerate(feats.columns):
        importances['Modelo'].extend([model] * 20)
        importances['Atributo'].extend([attribute] * 20)
        importances['Execução'].extend(range(20))
        importances['Importância'].extend(model_importances['importances'][i])

importances_df = pd.DataFrame(importances)
sns.barplot(x='Atributo', y='Importância', hue='Modelo', data=importances_df, ci='sd', errwidth=0.8)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importances.pdf')
