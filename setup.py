from setuptools import setup, find_packages
import os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))

    return paths

extra_files = package_files("models")

setup(
    name='grants-tagger',
    author='Nick Sorros',
    author_email='n.sorros@wellcome.ac.uk',
    description='A machine learning model to tag grants',
    packages=find_packages(),
    include_package_data=True,
    version='2020.2.0',
    package_data={'': extra_files},
    install_requires=[
        'pandas',
        'xlrd',
        'scikit-learn==0.21.3',
        'nltk',
        'matplotlib',
        'wellcomeml[deep-learning]'
    ],
    tests_require=[
        'pytest',
        'pytest-cov'
    ]
)
