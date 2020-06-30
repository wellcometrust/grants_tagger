# encoding: utf-8
"""
Run experiments with different approaches
"""
from argparse import ArgumentParser
from pathlib import Path
import os

from src.utils import load_data
from src.train import train_and_evaluate

if __name__ == '__main__':
    argparser = ArgumentParser(description=__doc__.strip())
    argparser.add_argument('-d', '--data', type=Path, help="path to processed JSON data to be used for training")
    argparser.add_argument('-l', '--label_binarizer', type=Path, help="path to label binarizer")
    argparser.add_argument('-e', '--experiment_name', type=str, help="name of experiment")
    args = argparser.parse_args()
    
    approaches = [
        'tfidf-svm',
        'spacy-textclassifier',
        'classifierchain-tfidf-svm',
        'labelpowerset-tfidf-svm',
        'binaryrelevance-tfidf-svm',
        'binaryrelevance-tfidf-knn',
        'tfidf-bert',
        'tfidf-scibert',
        'bert',
        'scibert'
    ]
    for approach in approaches:
        print(approach)
        model_dir = os.path.join(os.path.dirname(__file__), '../models')
        score = train_and_evaluate(
            args.data,
            args.label_binarizer,
            approach,
            model_path="{model_dir}/{approach}_{experiment_name}{extention}".format(
                model_dir=model_dir,
                approach=approach,
                experiment_name=args.experiment_name,
                extention=".pkl" if approach not in ['bert', 'scibert'] else "")
        )
        print(score)
