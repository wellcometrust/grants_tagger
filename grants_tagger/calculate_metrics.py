# encoding: utf-8
"""
This script aims to explore how the model performs in more detail
"""
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from pathlib import Path
import os.path
import pickle

from datascience.ml.bert_classifier import BertClassifier
from science_tagger.utils import load_data


def calculate_feature_importance_per_tag(model, tags, nb_features=5):
    # To do: This model does not generalise to non linear models
    #    so we need to move towards using rank features approaches

    try:
        # The model is a pipeline with the first step being the vectorizer
        vectorizer = model['tfidf']
        # And the second being the classifier
        clf = model['clf']

        # TODO: Make this work for Vectorizer
        vocabulary = vectorizer.vocabulary_
        coef = clf.coef_.todense()
    except Exception as e:
        print(e)
        # If not then return None for now
        return

    index_to_word = {v:k for k,v in vocabulary.items()}

    feature_importance_per_tag = []
    for tag_index, tag in enumerate(tags):
        features_coef = np.squeeze(np.asarray(coef[tag_index,:]))

        min_coef_indices = np.argsort(features_coef)[:nb_features]
        min_coef = features_coef[min_coef_indices].tolist()
        min_coef_features = [index_to_word[index] for index in min_coef_indices]

        max_coef_indices = np.flip(np.argsort(features_coef)[-nb_features:])
        max_coef = features_coef[max_coef_indices].tolist()
        max_coef_features = [index_to_word[index] for index in max_coef_indices]

        feature_coefs = max_coef + min_coef
        features = max_coef_features + min_coef_features

        for tag_coef, feature in zip(feature_coefs, features):
            feature_importance_per_tag.append((tag, tag_coef, feature))

    return pd.DataFrame(feature_importance_per_tag, columns=['Tag', 'Coef', 'Feat'])

def ml_roc_curve(Y, Y_score, nb_thresholds=10):
    """
    Input:
        Y: (nb_examples, nb_labels)
        Y_score: (nb_examples, nb_labels) probabilities for each example
    Output:
        precisions: (nb_thresholds, ) precision score for each threshold
        recalls: (nb_thresholds, ) recall score for each threshold
        thresholds: (nb_thresholds, ) list of thresholds (above which we assign label), default number of thresholds is 10
    """

    precisions = []
    recalls = []
    thresholds = np.linspace(0,0.95,nb_thresholds)
    for threshold in thresholds:
        Y_pred = (Y_score > threshold).astype(int)
        precisions.append(precision_score(Y_test, Y_pred, average = 'micro'))
        recalls.append(recall_score(Y_test, Y_pred, average = 'micro'))

    return precisions, recalls, thresholds

def avtag_thresh(Y_score, nb_thresholds=10):
    """
    Input: list of Y_scores

    Output:
        avtag: average number of tags for each threshold
        thresholds: (nb_thresholds, ) list of thresholds, default number of thresholds is 10
    """
    avtag = []
    thresholds = np.linspace(0,0.95,nb_thresholds)
    for threshold in thresholds:
        Y_pred_test = (Y_score >= threshold).astype(int)
        #Y_pred_test = model.predict(X_test)
        avtag.append(Y_pred_test.sum(axis=1).mean())

    return avtag, thresholds

def get_tagger_predictions(data, tagger_level, label_binarizer):
    tagger_tags_col = 'Tagger{}_tags'.format(tagger_level)
    return label_binarizer.transform([t[tagger_tags_col].split(',') for t in data])

def plot_human_accuracy_per_tag(metrics):
    metrics = metrics.sort_values(by=['n'])
    # reduce length of titles that are too long for plot
    metrics = metrics.replace('24: Data Science Computational & Mathematical Modelling', '24: Data Science/Modelling')
    metrics = metrics.replace('2: Gene Regulation and Genome Integrity', '2: Gene Reg. & Genome Integrity')
    metrics = metrics.replace('19: Maternal Child & Adolescent Health', '19: Maternal Child & Adol. Health')

    fig, ax1 = plt.subplots(figsize=(20,12))
    fig.subplots_adjust(bottom=0.35)
    metrics.plot(x ='Tag', y='human_f1', color = 'k', ax=ax1, fontsize = 14)
    metrics.plot(kind='bar', x ='Tag', y='model_f1', ax=ax1, fontsize = 14)
    ax1.set_ylim(0, 1)

    plt.savefig(os.path.join(os.path.dirname(__file__), '../figures/{}.png'.format('human_metrics')))

def plot_precision_recall_curve(tpr, fpr):
    fig, ax1 = plt.subplots(figsize=(12, 12))
    ax1.step(tpr, fpr)
    plt.ylabel("Precision", fontsize = 14)
    plt.xlabel("Recall", fontsize = 14)
    plt.savefig(os.path.join(os.path.dirname(__file__), '../figures/precisionvsrecall.png'))

def plot_tags_count_per_threhsold(Y_score):
    avtag, thresholds = avtag_thresh(Y_score, 20)
    fig, ax2 = plt.subplots(figsize=(12, 12))
    ax2.step(thresholds, avtag)
    plt.xlabel("Threshold", fontsize = 14)
    plt.ylabel("Average number of tags", fontsize = 14)
    plt.axhline(y = 2.3, color = 'r', linestyle = 'dashed')
    plt.savefig(os.path.join(os.path.dirname(__file__), '../figures/avtagperthreshold.png'))

def plot_tag_count_vs_accuracy(metrics):
    fig, ax1 = plt.subplots(figsize=(12,12))
    # add title below if required
    ax1.set_title(
        ''
    )
    ax1.set_xlabel('Number of examples', fontsize = 16)
    ax1.set_ylabel('f1 score', fontsize = 16)
    ax1.scatter(
        metrics['n'],
        metrics['model_f1'],
        s = 160
    )
    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)

    for x,y,tag in zip(metrics['n'], metrics['model_f1'], tags):

        label = "{:}".format(tag)

        plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center')

    plt.savefig(os.path.join(
        os.path.dirname(__file__),
        '../figures/tag_count_performance_scatter_points.png'
    ))


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--data', type=Path, help="Processed Json containing data")
    argparser.add_argument('--label_binarizer', type=Path, help="path to pickled binarizer for tags")
    argparser.add_argument('--model', type=Path, help="path to pickled scikit model")

    args = argparser.parse_args()

    with open(args.label_binarizer, 'rb') as f:
        label_binarizer = pickle.load(f)

    X, Y, meta = load_data(args.data, label_binarizer)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=42
    )
    tags = label_binarizer.classes_

    if os.path.isdir(args.model):
        model = BertClassifier()
        model.load(args.model)
    else:
        with open(args.model, 'rb') as f:
            model = pickle.load(f)

    Y_pred = model.predict(X_test)
    Y_score = model.predict_proba(X_test)

    Y_tagger_1 = get_tagger_predictions(meta, 1, label_binarizer)
    Y_tagger_2 = get_tagger_predictions(meta, 2, label_binarizer)

    feature_importances = calculate_feature_importance_per_tag(model, tags)
    print(feature_importances)

    metrics = pd.DataFrame({
        'Tag': tags,
        'human_f1': f1_score(Y, Y_tagger_1, average=None),
        'model_f1': f1_score(Y_test, Y_pred, average=None),
        'n': Y.sum(axis=0)
    })
    print(metrics)
    
    auc_score = roc_auc_score(Y_test, Y_score)
    print("AUC {:2f}".format(auc_score))

    print("Human micro f1 {:2f}".format(f1_score(Y, Y_tagger_1, average='micro')))
    print("Model micro f1 {:2f}".format(f1_score(Y_test, Y_pred, average='micro')))

    fpr, tpr, thresholds = ml_roc_curve(Y_test, Y_score, 12)

    plot_precision_recall_curve(tpr, fpr)
    plot_tags_count_per_threhsold(Y_score)
    plot_human_accuracy_per_tag(metrics)
    plot_tag_count_vs_accuracy(metrics)

