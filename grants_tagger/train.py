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
from sklearn.preprocessing import Normalizer, OneHotEncoder, FunctionTransformer
from scipy.sparse import hstack

import pickle
import os.path
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
    elif approach == 'tfidf+onehot_team-svm':
        model = Pipeline([
            ('vectorizer', FeatureUnion([
                ("text_features", Pipeline([
                    ("selector", FunctionTransformer(lambda x: x["text"])),
                    ("tfidf", TfidfVectorizer(min_df=5, ngram_range=(1,2), stop_words="english"))
                ])),
                ("team_features", Pipeline([
                    ("selector", FunctionTransformer(lambda x: x[["Team"]])),
                    ("one hot", OneHotEncoder(handle_unknown='ignore'))
                ]))
            ])),
            ('svm', OneVsRestClassifier(SVC(class_weight="balanced", kernel="linear")))
        ])
    elif approach == 'tfidf+onehot_scheme-svm':
        model = Pipeline([
            ('vectorizer', FeatureUnion([
                ("text_features", Pipeline([
                    ("selector", FunctionTransformer(lambda x: x["text"])),
                    ("tfidf", TfidfVectorizer(min_df=5, ngram_range=(1,2), stop_words="english"))
                ])),
                ("team_features", Pipeline([
                    ("selector", FunctionTransformer(lambda x: x[["Scheme"]])),
                    ("one hot", OneHotEncoder(handle_unknown='ignore'))
                ]))
            ])),
            ('svm', OneVsRestClassifier(SVC(class_weight="balanced", kernel="linear")))
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
        y_batch_size=None, X_format="List",
        test_size=0.25, verbose=True):

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
            from_same_distribution=from_same_distribution,
            X_format=X_format,
            test_size=test_size
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
        if y_batch_size:
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
        if str(model_path).endswith('pkl') or str(model_path).endswith('pickle'):
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        elif y_batch_size: 
            pass # saved already
        else: # dir path that might not exist
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            if approach in ["cnn", "bilstm"]:
                # cnn and bilstm are pipelines so do not have a save
                # but CNNClassifier and BiLSTMClassifier do have one
                vectorizer = model.steps[0][1]
                classifier = model.steps[1][1]

                vectorizer_path = os.path.join(model_path, "vectorizer.pkl")
                with open(vectorizer_path, "wb") as f:
                    f.write(pickle.dumps(vectorizer))
                classifier.save(model_path)

            else: # default behaviour assumes that model has a save if path given
                model.save(model_path)

    return f1
