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
        'matplotlib',
        'wellcomeml[tensorflow]==1.2.0',
        'docutils==0.15',
        'scipy==1.4.1',
        'wasabi',
        'typer',
        'scispacy',
        'tqdm',
        'streamlit',
        'requests',
        'pygtrie==2.3.3'
    ],
    tests_require=[
        'pytest',
        'pytest-cov',
        'openpyxl'
    ]
)
