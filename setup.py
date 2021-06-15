from setuptools import setup, find_packages
import os

from grants_tagger import __version__


setup(
    name='grants-tagger',
    author='Nick Sorros',
    author_email='n.sorros@wellcome.ac.uk',
    description='A machine learning model to tag grants',
    packages=find_packages(),
    version=__version__,
    entry_points = {
        'console_scripts': 'grants_tagger=grants_tagger.__main__:app'
    },
    install_requires=[
        'pandas',
        'xlrd',
        'scikit-learn==0.23.2',
        'nltk',
        'matplotlib',
        'wellcomeml[deep-learning]==1.0.2',
        'docutils==0.15',
        'scipy==1.4.1',
        'wasabi',
        'typer',
        'scispacy',
        'dvc',
        'tqdm'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'openpyxl'
    ]
)
