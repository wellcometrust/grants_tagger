from setuptools import setup, find_packages

from grants_tagger.__version__ import __version__

setup(
    name="grants-tagger",
    author="Nick Sorros",
    author_email="n.sorros@wellcome.ac.uk",
    description="A machine learning model to tag grants",
    packages=find_packages(),
    version=__version__,
    entry_points={"console_scripts": "grants_tagger=grants_tagger.__main__:app"},
    install_requires=["sklearn", "pandas", "numpy", "scipy", "requests"],
    extras_require={
        "xlinear": ["libpecos==0.1.0"],
        "dev": [
            "pandas",
            "xlrd",
            "scikit-learn==0.23.2",
            "matplotlib",
            "wellcomeml[tensorflow,core,transformers,torch]==2.0.3",
            "docutils==0.15",
            "scipy==1.4.1",
            "wasabi",
            "typer",
            "tqdm",
            "streamlit",
            "requests",
            "pygtrie==2.3.3",
            "openpyxl",
        ],
    },
    tests_require=["pytest", "tox"],
)
