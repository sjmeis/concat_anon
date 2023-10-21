# *Lexical Substitution is not Synonym Substitution: On the Importance of Producing Contextually Relevant Word Substitutes*
## Software
Files in this directory:

- `ls.ipynb`: notebook for performing LS on the dataset included below
- `ls_train.ipynb`: used to train models on the LS replaced data
- `qualitative.ipynb`: code to replicate our qualitative analysis performed in the paper
- `score.pl`: scoring script for the three benchmarks
- `concat/*`: LS prediction files for use with the scoring script above (includes gold files)
- `results/*`: LS prediction files for use in `qualitative.ipynb`
- `util/train.py`: util for model traning using keras
- `util/wordvec_load.py`: util for loading GloVe embeddings

Data files:

- `data/ag_news_preprocessed_train.csv`

Note that the appropropriate GloVe embedding files can be found at: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

In order to install all Python dependencies, please run: `pip install -r requirements.txt`