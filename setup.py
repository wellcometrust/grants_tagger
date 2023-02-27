from setuptools import setup, find_packages

from grants_tagger.__version__ import __version__

setup(
    name="grants-tagger",
    author="Arne Robben",
    author_email="a.robben@wellcome.ac.uk",
    long_description="A machine learning model to tag grants. You can find more information on https://github.com/wellcometrust/grants_tagger",
    packages=find_packages(),
    version=__version__,
    entry_points={"console_scripts": "grants_tagger=grants_tagger.__main__:app"},
    install_requires=[
        "scikit-learn==1.0.0",
        "libpecos==0.3.0",
        "typer",
        "pandas",
        "numpy",
        "scipy",
        "requests",
        "dvc",
        "wasabi",
        "wellcomeml",
        "gensim==4.0.0",
    ],
    tests_require=["pytest", "tox"],
)
