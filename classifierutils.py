import random
import re
from typing import *

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec, doc2vec
from nltk.corpus import stopwords
from nltk.tokenize.regexp import regexp_tokenize
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from mytypes import *

tqdm.pandas(desc="progress-bar")

stopwords = set(stopwords.words('english'))


def corpus_preprocess(corpus: pd.Series, header: str) -> Tuple[List[SampleLabelled], List[SampleLabelledTokenised]]:
    """Pre-process data and format them in structure that is easy to pass to 
    other methods

    Parameters
    ----------
    sample : str
        The sample you want to process
    header : str
        The label you want to apply to the sample
    """
    corpus_labelled = []
    corpus_labelled_tokenised = []

    for sample in corpus:
        sample_preprocessed = sample_preprocess(sample, header)
        corpus_labelled.append(sample_preprocessed[0])
        corpus_labelled_tokenised.append(sample_preprocessed[1])

    return pd.DataFrame(corpus_labelled, columns=['TEXT', 'HEADER']), pd.DataFrame(corpus_labelled_tokenised, columns=['TEXT', 'HEADER'])


def sample_preprocess(sample: str, header: str) -> Tuple[SampleLabelled, SampleLabelledTokenised]:
    """Pre-process data and format them in structure that is easy to pass to 
    other methods

    Parameters
    ----------
    sample : str
        The sample you want to process
    header : str
        The label you want to apply to the sample
    """

    # """Clean sample"""
    temp = sample.lower()

    # Remove all occurences of square brackets and everything in between
    temp = re.sub("\[.*?\]", "", temp)

    # Focus on words, disregard numbers, etc.
    temp_tokens = regexp_tokenize(temp, r"[a-zA-z]+")

    # Remove stop words
    temp_tokens = [word for word in temp_tokens if not word in stopwords]

    # Create labelled sample
    sample_labelled = (' '.join(temp_tokens), header)

    # Create labelled tokenised sample
    sample_labelled_tokenised = (temp_tokens, header)

    return sample_labelled, sample_labelled_tokenised
