from pathlib import Path
import logging
import ast

logger = logging.getLogger(__name__)

try:
    from skmultilearn.problem_transform import (
        ClassifierChain,
        LabelPowerset,
        BinaryRelevance,
    )
    from skmultilearn.adapt import BRkNNaClassifier, MLkNN
except ModuleNotFoundError as e:
    logger.warning(
        "skmultilearn not installed. laber powerset, classifier chain and binary relevance approaches not working"
    )
    logger.debug(e)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,
)
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer, OneHotEncoder, FunctionTransformer

from grants_tagger.models.mesh_cnn import MeshCNN
from grants_tagger.models.tfidf_transformers_svm import TfidfTransformersSVM
from grants_tagger.models.science_ensemble import ScienceEnsemble
from grants_tagger.models.mesh_tfidf_svm import MeshTfidfSVM
from grants_tagger.models.bert_mesh import WellcomeBertMesh
from grants_tagger.utils import save_pickle, load_pickle
from wellcomeml.ml.bilstm import BiLSTMClassifier
from wellcomeml.ml.cnn import CNNClassifier
from wellcomeml.ml.keras_vectorizer import KerasVectorizer
from wellcomeml.ml.bert_classifier import BertClassifier
from wellcomeml.ml.voting_classifier import WellcomeVotingClassifier
from wellcomeml.ml.transformers_tokenizer import TransformersTokenizer
from wellcomeml.ml.sent2vec_vectorizer import Sent2VecVectorizer
from wellcomeml.ml.doc2vec_vectorizer import Doc2VecVectorizer
from wellcomeml.ml.bert_vectorizer import BertVectorizer

try:
    from wellcomeml.ml.spacy_classifier import SpacyClassifier
except ImportError as e:
    logger.warning("SpacyClassifier missing. Approach spacy-classifier not working")
    logger.debug(e)
try:
    from grants_tagger.models.mesh_xlinear import MeshXLinear
except ImportError:
    logger.warning(
        "pecos library is not installed possbly because you are not in a Linux machine. XLinear will not work"
    )


class ApproachNotImplemented(Exception):
    pass


def create_model(approach, parameters=None):
    if approach == "tfidf-svm":
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        max_df=0.95,
                        min_df=0.0,
                        ngram_range=(1, 1),
                    ),
                ),
                ("svm", OneVsRestClassifier(SVC(kernel="linear", probability=True))),
            ]
        )
    elif approach == "tfidf-transformers-svm":
        model = TfidfTransformersSVM()
    elif approach == "bert-svm":
        model = Pipeline(
            [
                ("bert", BertVectorizer(pretrained="bert")),
                ("svm", OneVsRestClassifier(SVC(kernel="linear", probability=True))),
            ]
        )
    elif approach == "scibert-svm":
        model = Pipeline(
            [
                ("scibert", BertVectorizer(pretrained="scibert")),
                ("svm", OneVsRestClassifier(SVC(kernel="linear", probability=True))),
            ]
        )
    elif approach == "spacy-textclassifier":
        model = SpacyClassifier()
    elif approach == "bert":
        model = BertClassifier()
    elif approach == "scibert":
        model = BertClassifier(pretrained="scibert")
    elif approach == "classifierchain-tfidf-svm":
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        max_df=0.95,
                        min_df=0.0,
                        ngram_range=(1, 1),
                    ),
                ),
                (
                    "svm",
                    ClassifierChain(classifier=SVC(kernel="linear", probability=True)),
                ),
            ]
        )
    elif approach == "labelpowerset-tfidf-svm":
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        max_df=0.95,
                        min_df=0.0,
                        ngram_range=(1, 1),
                    ),
                ),
                ("svm", LabelPowerset(SVC(kernel="linear", probability=True))),
            ]
        )
    elif approach == "binaryrelevance-tfidf-svm":
        # same as OneVsRestClassifier
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        max_df=0.95,
                        min_df=0.0,
                        ngram_range=(1, 1),
                    ),
                ),
                (
                    "svm",
                    BinaryRelevance(classifier=SVC(kernel="linear", probability=True)),
                ),
            ]
        )
    elif approach == "binaryrelevance-tfidf-knn":
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english",
                        max_df=0.95,
                        min_df=0.0,
                        ngram_range=(1, 1),
                    ),
                ),
                ("knn", BinaryRelevance(classifier=KNeighborsClassifier)),
            ]
        )
    elif approach == "hashing_vectorizer-svm":
        model = Pipeline(
            [
                ("hashing_vectorizer", HashingVectorizer()),
                ("svm", OneVsRestClassifier(SGDClassifier(loss="hinge", penalty="l2"))),
            ]
        )
    elif approach == "hashing_vectorizer-nb":
        model = Pipeline(
            [
                (
                    "hashing_vectorizer",
                    HashingVectorizer(binary=True, n_features=2**18),
                ),
                ("nb", OneVsRestClassifier(MultinomialNB())),
            ]
        )
    elif approach == "tfidf-sgd":
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        stop_words="english", max_df=0.95, min_df=5, ngram_range=(1, 1)
                    ),
                ),
                ("svm", OneVsRestClassifier(SGDClassifier(loss="hinge", penalty="l2"))),
            ]
        )
    elif approach == "cnn":
        model = Pipeline(
            [
                ("vec", KerasVectorizer(vocab_size=5_000)),
                (
                    "cnn",
                    CNNClassifier(
                        learning_rate=0.01,
                        dropout=0.1,
                        nb_epochs=20,
                        nb_layers=4,
                        multilabel=True,
                    ),
                ),
            ]
        )
    elif approach == "bilstm":
        model = Pipeline(
            [
                ("vec", KerasVectorizer(vocab_size=5_000, sequence_length=678)),
                (
                    "bilstm",
                    BiLSTMClassifier(
                        learning_rate=0.01, dropout=0.1, nb_epochs=20, multilabel=True
                    ),
                ),
            ]
        )
    elif approach == "doc2vec-sgd":
        model = Pipeline(
            [
                ("vec", Doc2VecVectorizer()),
                (
                    "sgd",
                    OneVsRestClassifier(
                        SGDClassifier(penalty="l2", alpha=1e-8), n_jobs=-1
                    ),
                ),
            ]
        )
    elif approach == "doc2vec-tfidf-sgd":
        model = Pipeline(
            [
                (
                    "vec",
                    FeatureUnion(
                        [
                            (
                                "doc2vec",
                                Pipeline(
                                    [
                                        ("doc2vec_unscaled", Doc2VecVectorizer()),
                                        ("scale_doc2vec", Normalizer()),
                                    ]
                                ),
                            ),
                            (
                                "tfidf",
                                Pipeline(
                                    [
                                        (
                                            "tfidf_unscaled",
                                            TfidfVectorizer(
                                                min_df=5,
                                                stop_words="english",
                                                ngram_range=(1, 2),
                                            ),
                                        ),
                                        ("scale_tfidf", Normalizer()),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "sgd",
                    OneVsRestClassifier(
                        SGDClassifier(penalty="l2", alpha=1e-6), n_jobs=-1
                    ),
                ),
            ]
        )
    elif approach == "sent2vec-sgd":
        model = Pipeline(
            [
                ("vec", Sent2VecVectorizer(pretrained="biosent2vec")),
                (
                    "sgd",
                    OneVsRestClassifier(
                        SGDClassifier(penalty="l2", alpha=1e-8), n_jobs=-1
                    ),
                ),
            ]
        )
    elif approach == "sent2vec-tfidf-sgd":
        model = Pipeline(
            [
                (
                    "vec",
                    FeatureUnion(
                        [
                            (
                                "sent2vec",
                                Pipeline(
                                    [
                                        (
                                            "sent2vec_unscaled",
                                            Sent2VecVectorizer(
                                                pretrained="biosent2vec"
                                            ),
                                        ),
                                        ("scale_sent2vec", Normalizer()),
                                    ]
                                ),
                            ),
                            (
                                "tfidf",
                                Pipeline(
                                    [
                                        (
                                            "tfidf_unscaled",
                                            TfidfVectorizer(
                                                min_df=5,
                                                stop_words="english",
                                                ngram_range=(1, 2),
                                            ),
                                        ),
                                        ("scale_tfidf", Normalizer()),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "sgd",
                    OneVsRestClassifier(
                        SGDClassifier(penalty="l2", alpha=1e-8), n_jobs=-1
                    ),
                ),
            ]
        )
    elif approach == "tfidf-adaboost":
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(min_df=5, stop_words="english", ngram_range=(1, 2)),
                ),
                (
                    "adaboost",
                    OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier())),
                ),
            ]
        )
    elif approach == "tfidf-gboost":
        model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(min_df=5, stop_words="english", ngram_range=(1, 2)),
                ),
                ("gboost", OneVsRestClassifier(GradientBoostingClassifier())),
            ]
        )
    elif approach == "tfidf+onehot_team-svm":
        model = Pipeline(
            [
                (
                    "vectorizer",
                    FeatureUnion(
                        [
                            (
                                "text_features",
                                Pipeline(
                                    [
                                        (
                                            "selector",
                                            FunctionTransformer(lambda x: x["text"]),
                                        ),
                                        (
                                            "tfidf",
                                            TfidfVectorizer(
                                                min_df=5,
                                                ngram_range=(1, 2),
                                                stop_words="english",
                                            ),
                                        ),
                                    ]
                                ),
                            ),
                            (
                                "team_features",
                                Pipeline(
                                    [
                                        (
                                            "selector",
                                            FunctionTransformer(lambda x: x[["Team"]]),
                                        ),
                                        (
                                            "one hot",
                                            OneHotEncoder(handle_unknown="ignore"),
                                        ),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "svm",
                    OneVsRestClassifier(SVC(class_weight="balanced", kernel="linear")),
                ),
            ]
        )
    elif approach == "tfidf+onehot_scheme-svm":
        model = Pipeline(
            [
                (
                    "vectorizer",
                    FeatureUnion(
                        [
                            (
                                "text_features",
                                Pipeline(
                                    [
                                        (
                                            "selector",
                                            FunctionTransformer(lambda x: x["text"]),
                                        ),
                                        (
                                            "tfidf",
                                            TfidfVectorizer(
                                                min_df=5,
                                                ngram_range=(1, 2),
                                                stop_words="english",
                                            ),
                                        ),
                                    ]
                                ),
                            ),
                            (
                                "team_features",
                                Pipeline(
                                    [
                                        (
                                            "selector",
                                            FunctionTransformer(
                                                lambda x: x[["Scheme"]]
                                            ),
                                        ),
                                        (
                                            "one hot",
                                            OneHotEncoder(handle_unknown="ignore"),
                                        ),
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                (
                    "svm",
                    OneVsRestClassifier(SVC(class_weight="balanced", kernel="linear")),
                ),
            ]
        )
    elif approach == "mesh-tfidf-svm":
        model = MeshTfidfSVM()
    elif approach == "mesh-cnn":
        model = MeshCNN()
    elif approach == "science-ensemble":
        model = ScienceEnsemble()
    elif approach == "mesh-xlinear":
        model = MeshXLinear()
    elif approach == "bertmesh":
        model = WellcomeBertMesh()
    else:
        raise ApproachNotImplemented
    if parameters:
        params = ast.literal_eval(parameters)
        model.set_params(**params)
    else:
        parameters = {}
    return model


def load_model(approach, model_path, parameters=None):
    if str(model_path).endswith(".pkl"):
        model = load_pickle(model_path)
    else:
        model = create_model(approach, parameters=parameters)
        model.load(model_path)

    return model
