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
    version='2020.11.0',
    package_data={'': extra_files},
    entry_points = {
        'console_scripts': 'grants_tagger=grants_tagger.__main__:app'
    },
    install_requires=[
        'pandas',
        'xlrd',
        'scikit-learn==0.23.2',
        'nltk',
        'matplotlib',
        'wellcomeml[deep-learning]==2021.2.1',
        'docutils==0.15',
        'scipy==1.4.1',
        'wasabi',
        'typer'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'openpyxl'
    ]
)
