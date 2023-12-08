from setuptools import setup, find_packages
from setuptools.command.install import install
# from distutils.sysconfig import get_python_lib
import spacy
import nltk
from pathlib import Path
from dutch_text_analytics import __version__


with Path("README.md").open(encoding="utf-8") as f:
    long_description = f.read()

class CustomInstallCommand(install):
    def run(self):
        # Run the default installation
        install.run(self)

        # Download spaCy models
        spacy.cli.download("nl_core_news_lg")
        spacy.cli.download("en_core_web_lg")
        spacy.cli.download("nl_core_news_sm")
        spacy.cli.download("en_core_web_sm")

        # Download NLTK data
        nltk.download('punkt')

setup(
    name='dutch_text_analytics',
    version=__version__,
    description='Dutch Text Analytics is a versatile toolkit designed to facilitate the exploration, execution, and validation of a diverse range of Natural Language Processing (NLP) tasks specifically tailored for the Dutch language. This repository provides a comprehensive set of tools, including code examples, scripts, and resources, to enhance and streamline your Dutch NLP projects.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'spacy>=3.0.0',          # Update to the latest version of spacy
        'gensim>=4.0.0',         # Update to the latest version of gensim
        'ipywidgets>=7.6.0',     # Update to the latest version of ipywidgets
        'jupyter>=1.0.0',        # Update to the latest version of jupyter
        'keybert>=0.4.0',        # Update to the latest version of keybert
        'pandas>=1.0.0',         # Update to the latest version of pandas
        'numpy>=1.18.0',         # Update to the latest version of numpy
        'scipy>=1.4.0',          # Update to the latest version of scipy
        'scikit-learn>=0.24.0',  # Update to the latest version of scikit-learn
        'nltk>=3.5.0',           # Update to the latest version of nltk
        'transformers>=4.5.0',   # Update to the latest version of transformers
        'torch>=1.8.0',          # Update to the latest version of torch
        'openpyxl>=3.0.0',       # Update to the latest version of openpyxl
        'python-Levenshtein>=0.12.0',  # Update to the latest version of python-Levenshtein
        'sent2vec>=0.2.0',       # Update to the latest version of sent2vec
        'sentence-transformers>=2.0.0'  # Update to the latest version of sentence-transformers
    ],
    cmdclass={'install': CustomInstallCommand},
)