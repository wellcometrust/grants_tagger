from setuptools import setup, find_packages

setup(
    name='grants-tagger',
    author='Nick Sorros',
    author_email='n.sorros@wellcome.ac.uk',
    description='A machine learning model to tag grants',
    packages=find_packages(),
    include_package_data=True,
    version='2020.2.0',
    install_requires=[
        'pandas',
        'xlrd',
        'scikit-learn==0.21.3',
        'nltk',
        'matplotlib',
        'wellcomeml[deep-learning]'
     ]
)
