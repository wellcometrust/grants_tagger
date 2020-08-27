from setuptools import setup, find_packages
import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))

    return paths

extra_files = package_files("models/scibert-2020.05.5") + [
    "../models/tfidf-svm-2020.05.2.pkl",
    "../models/label_binarizer.pkl"
]

setup(
    name='grants-tagger',
    author='Nick Sorros',
    author_email='n.sorros@wellcome.ac.uk',
    description='A machine learning model to tag grants',
    packages=find_packages(),
    include_package_data=True,
    version='2020.07.2',
    package_data={'': extra_files},
    install_requires=[
        'pandas',
        'xlrd',
        'scikit-learn==0.21.3',
        'nltk',
        'matplotlib',
        'wellcomeml[deep-learning]==2020.07.0',
        'docutils==0.15',
        'scipy==1.4.1'
    ],
    tests_require=[
        'pytest',
        'pytest-cov'
    ]
)
