# encoding: utf-8
"""
Train a spacy or sklearn model and pickle it
"""
from skmultilearn.problem_transform import ClassifierChain, LabelPowerset, BinaryRelevance
from skmultilearn.adapt import BRkNNaClassifier, MLkNN
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import Normalizer
from scipy.sparse import hstack

from argparse import ArgumentParser
from distutils.util import strtobool
from pathlib import Path
import configparser
import pickle
import os.path
import json
import ast

from wellcomeml.ml import BiLSTMClassifier, CNNClassifier, KerasVectorizer, SpacyClassifier, BertVectorizer, BertClassifier, Doc2VecVectorizer, Sent2VecVectorizer
from grants_tagger.utils import load_train_test_data, yield_train_data, load_test_data


class ApproachNotImplemented(Exception):
    pass

def create_model(approach, parameters=None):
    if approach == 'tfidf-svm':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', max_df=0.95,
                min_df=0.0, ngram_range=(1,1)
            )),
            ('svm', OneVsRestClassifier(SVC(kernel='linear', probability=True)))
        ])
    elif approach == 'bert-svm':
        model = Pipeline([
            ('bert', BertVectorizer(pretrained='bert')),
            ('svm', OneVsRestClassifier(SVC(kernel='linear', probability=True)))
        ])
    elif approach == 'scibert-svm':
        model = Pipeline([
            ('scibert', BertVectorizer(pretrained='scibert')),
            ('svm', OneVsRestClassifier(SVC(kernel='linear', probability=True)))
        ])
    elif approach == 'spacy-textclassifier':
        model = SpacyClassifier()
    elif approach == 'bert':
        model = BertClassifier()
    elif approach == 'scibert':
        model = BertClassifier(pretrained='scibert')
    elif approach == 'classifierchain-tfidf-svm':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', max_df=0.95,
                min_df=0.0, ngram_range=(1,1)
            )),
            ('svm', ClassifierChain(classifier=SVC(kernel='linear', probability=True)))
        ])
    elif approach == 'labelpowerset-tfidf-svm':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', max_df=0.95,
                min_df=0.0, ngram_range=(1,1)
            )),
            ('svm', LabelPowerset(SVC(kernel='linear', probability=True)))
        ])
    elif approach == 'binaryrelevance-tfidf-svm':
        # same as OneVsRestClassifier
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', max_df=0.95,
                min_df=0.0, ngram_range=(1,1)
            )),
            ('svm', BinaryRelevance(classifier=SVC(kernel='linear', probability=True)))
        ])
    elif approach == 'binaryrelevance-tfidf-knn':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', max_df=0.95,
                min_df=0.0, ngram_range=(1,1)
            )),
            ('knn', BinaryRelevance(classifier=KNeighborsClassifier))
        ])
    elif approach == 'hashing_vectorizer-svm':
        model = Pipeline([
            ('hashing_vectorizer', HashingVectorizer()),
            ('svm', OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2')))
        ])
    elif approach == 'hashing_vectorizer-nb':
        model = Pipeline([
            ('hashing_vectorizer', HashingVectorizer(binary=True, n_features=2 ** 18)),
            ('nb', OneVsRestClassifier(MultinomialNB()))
        ])
    elif approach == 'tfidf-sgd':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english', max_df=0.95,
                min_df=5, ngram_range=(1,1)
            )),
            ('svm', OneVsRestClassifier(SGDClassifier(loss='hinge', penalty='l2')))
        ])
    elif approach == 'cnn':
        model = Pipeline([
            ('vec', KerasVectorizer(vocab_size=5_000)),
            ('cnn', CNNClassifier(learning_rate=0.01, dropout=0.1, nb_epochs=20, nb_layers=4, multilabel=True))
        ])
    elif approach == 'bilstm':
        model = Pipeline([
            ('vec', KerasVectorizer(vocab_size=5_000, sequence_length=678)),
            ('bilstm', BiLSTMClassifier(learning_rate=0.01, dropout=0.1, nb_epochs=20, multilabel=True))
        ])
    elif approach == 'doc2vec-sgd':
        model = Pipeline([
            ('vec', Doc2VecVectorizer()),
            ('sgd', OneVsRestClassifier(SGDClassifier(penalty="l2", alpha=1e-8), n_jobs=-1))
        ])
    elif approach == 'doc2vec-tfidf-sgd':
        model = Pipeline([
            ('vec', FeatureUnion([
                ('doc2vec', Pipeline([
                    ('doc2vec_unscaled', Doc2VecVectorizer()),
                    ('scale_doc2vec', Normalizer())
                ])),
                ('tfidf', Pipeline([
                    ('tfidf_unscaled', TfidfVectorizer(min_df=5, stop_words='english', ngram_range=(1,2))),
                    ('scale_tfidf', Normalizer())
                ]))
            ])),
            ('sgd', OneVsRestClassifier(SGDClassifier(penalty="l2", alpha=1e-6), n_jobs=-1))
        ])
    elif approach == 'sent2vec-sgd':
        model = Pipeline([
            ('vec', Sent2VecVectorizer(pretrained="biosent2vec")),
            ('sgd', OneVsRestClassifier(SGDClassifier(penalty="l2", alpha=1e-8), n_jobs=-1))
        ])
    elif approach == 'sent2vec-tfidf-sgd':
        model = Pipeline([
            ('vec', FeatureUnion([
                ('sent2vec', Pipeline([
                    ('sent2vec_unscaled', Sent2VecVectorizer(pretrained="biosent2vec")),
                    ('scale_sent2vec', Normalizer())
                ])),
                ('tfidf', Pipeline([
                    ('tfidf_unscaled', TfidfVectorizer(min_df=5, stop_words='english', ngram_range=(1,2))),
                    ('scale_tfidf', Normalizer())
                ]))
            ])),
            ('sgd', OneVsRestClassifier(SGDClassifier(penalty="l2", alpha=1e-8), n_jobs=-1))
        ])
    elif approach == 'tfidf-adaboost':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=5, stop_words='english', ngram_range=(1,2))),
            ('adaboost', OneVsRestClassifier(AdaBoostClassifier(DecisionTreeClassifier())))
        ])
    elif approach == 'tfidf-gboost':
        model = Pipeline([
            ('tfidf', TfidfVectorizer(min_df=5, stop_words='english', ngram_range=(1,2))),
            ('gboost', OneVsRestClassifier(GradientBoostingClassifier()))
        ])
    else:
        raise ApproachNotImplemented
    if parameters:
        params = ast.literal_eval(parameters)
        model.set_params(**params)
    else:
        parameters = {}
    return model

def train_and_evaluate(
        train_data_path, label_binarizer_path, approach,
        parameters=None, model_path=None, test_data_path=None,
        online_learning=False, nb_epochs=5,
        from_same_distribution=False, threshold=None,
        y_batch_size=None, verbose=True):

    with open(label_binarizer_path, "rb") as f:
        label_binarizer = pickle.load(f)

    # model creation from approach could move outside this function
    # so that run experiments can pass a model here
    model = create_model(approach, parameters)

    if online_learning:    
        X_test, Y_test = load_test_data(test_data_path, label_binarizer)
    
        # Note that not all online learning methods need multiple passes
        for epoch in range(nb_epochs):
            print(f"Epoch {epoch}")
            batch = 0
            for X_train, Y_train in yield_train_data(train_data_path, label_binarizer):
                batch_size = len(X_train)
                print(f"Batch {batch} - batch_size {batch_size}")
                # Since partial fit does not work in a Pipeline
                #   we assume two steps vectorizer and model to break down
                if isinstance(model, Pipeline):
                    vectorizer = model.steps[0][1]
                    classifier = model.steps[1][1]
                
                    vectorizer.partial_fit(X_train)
                    X_train = vectorizer.transform(X_train)
                else:
                    classifier = model

                classifier.partial_fit(X_train, Y_train, classes=list(range(len(Y_train[0]))))
                batch += 1
            Y_pred_test = model.predict(X_test)
            f1 = f1_score(Y_test, Y_pred_test, average='micro')
            print(f"f1: {f1}")
    else:
        X_train, X_test, Y_train, Y_test = load_train_test_data(
            train_data_path=train_data_path,
            test_data_path=test_data_path,
            label_binarizer=label_binarizer,
            from_same_distribution=from_same_distribution
        )
        if y_batch_size:
            vectorizer = model.steps[0][1]
            classifier = model.steps[1][1]

            Path(model_path).mkdir(exist_ok=True)
            print("Fitting vectorizer")
            vectorizer.fit(X_train)
            with open(f"{model_path}/vectorizer.pkl", "wb") as f:
                f.write(pickle.dumps(vectorizer))
            print("Training model")
            for tag_i in range(0, Y_train.shape[1], y_batch_size):
                print(tag_i)
                X_train_vec = vectorizer.transform(X_train)
                classifier.fit(X_train_vec, Y_train[:,tag_i:tag_i+y_batch_size]) # assuming that fit clears previous params
                with open(f"{model_path}/{tag_i}.pkl", "wb") as f:
                    f.write(pickle.dumps(classifier))
        else:
            model.fit(X_train, Y_train)

    if threshold:
        if y_batch_size:
            pass # to be implemented
        else:
            Y_pred_prob = model.predict_proba(X_test)
        Y_pred_test = Y_pred_prob > threshold
    else:
        if Y_test.shape[1] > y_batch_size:
            Y_pred_test = []
            for tag_i in range(0, Y_test.shape[1], y_batch_size):
                with open(f"{model_path}/{tag_i}.pkl", "rb") as f:
                    classifier = pickle.loads(f.read())
                X_test_vec = vectorizer.transform(X_test)
                Y_pred_test_i = classifier.predict(X_test_vec)
                Y_pred_test.append(Y_pred_test_i)
            Y_pred_test = hstack(Y_pred_test)
        else:
            Y_pred_test = model.predict(X_test)
            # Y_pred_train = model.predict(X_train)

    f1 = f1_score(Y_test, Y_pred_test, average='micro')
    if verbose:
        report = classification_report(Y_test, Y_pred_test, target_names=label_binarizer.classes_)
        print(report)

    if model_path:
        if ('pkl' in str(model_path)) or ('pickle' in str(model_path)):
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        elif y_batch_size: 
            pass # saved already
        else:
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model.save(model_path)

    return f1

if __name__ == "__main__":
    argparser = ArgumentParser(description=__doc__.strip())
    argparser.add_argument('-d', '--data', type=Path, help="path to processed JSON data to be used for training")
    argparser.add_argument('-l', '--label_binarizer', type=Path, help="path to label binarizer")
    argparser.add_argument('-m', '--model_path', type=Path, help="path to output pickled trained model")
    argparser.add_argument('-a', '--approach', type=str, default="tfidf-svm", help="tfidf-svm, bert-svm")
    argparser.add_argument('-p', '--parameters', type=str, help="model params in sklearn format e.g. {'svm__kernel: linear'}")
    argparser.add_argument('-t', '--test_data', type=Path, default=None, help="path to processed JSON test data")
    argparser.add_argument('-c', '--config', type=Path, help="path to config file with params set")
    argparser.add_argument('-o', '--online_learning', type=bool, default=False, help="flag to train in an online way")
    argparser.add_argument('-n', '--nb_epochs', type=int, default=5, help="number of passes of training data in online training")
    argparser.add_argument('-f', '--from_same_distribution', type=strtobool, default=False, help="whether train and test contain the same examples but differ in other ways, important when loading train and test parts of datasets")
    argparser.add_argument('--threshold', type=float, default=None, help="threhsold to assign a tag")
    argparser.add_argument('--y_batch_size', type=int, default=None, help="batch size for Y in cases where Y large. defaults to None i.e. no batching of Y")
    args = argparser.parse_args()

    if args.config:
        cfg = configparser.ConfigParser(allow_no_value=True)
        cfg.read(args.config)

        data_path = cfg["data"]["train_data_path"]
        label_binarizer_path = cfg["model"]["label_binarizer_path"]
        approach = cfg["model"]["approach"]
        parameters = cfg["model"]["parameters"]
        model_path = cfg["model"]["model_path"]
        test_data_path = cfg["data"]["test_data_path"]
        online_learning = bool(cfg["model"].get("online_learning", False))
        nb_epochs = int(cfg["model"].get("nb_epochs", 5))
        from_same_distribution = bool(cfg["data"].get("from_same_distribution", False))
        threshold = cfg["model"].get("threshold", None)
        if threshold:
            threshold = float(threshold)
        y_batch_size = int(cfg["model"].get("y_batch_size"))
    else:
        data_path = args.data
        label_binarizer_path = args.label_binarizer
        approach = args.approach
        parameters = args.parameters
        model_path = args.model_path
        test_data_path = args.test_data
        online_learning = args.online_learning
        nb_epochs = args.nb_epochs
        from_same_distribution = args.from_same_distribution
        threshold= args.threshold
        y_batch_size = args.y_batch_size

    if os.path.exists(model_path):
        print(f"{model_path} exists. Remove if you want to rerun.")
    else:
        train_and_evaluate(
            data_path,
            label_binarizer_path,
            approach,
            parameters,
            model_path=model_path,
            test_data_path=test_data_path,
            online_learning=online_learning,
            nb_epochs=nb_epochs,
            from_same_distribution=from_same_distribution,
            threshold=threshold,
            y_batch_size=y_batch_size
        )
