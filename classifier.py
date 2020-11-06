import random

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas(desc="progress-bar")
import re

from gensim.models import Doc2Vec, doc2vec
from sklearn import utils
from sklearn.utils import shuffle

import classifierutils
import dataread

headers = dataread.read_file('top_sectionheaders_5000.txt')

results = []

for header in headers:
    ret_val = []
    for item in header_corpus[header]['labelled_tokenised'].TEXT:
        logreg.predict([model_dbow.infer_vector(item)])
