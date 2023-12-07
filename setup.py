from setuptools import setup, find_packages
from setuptools.command.install import install
from distutils.sysconfig import get_python_lib
import spacy
import nltk

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
    version='0.1',
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