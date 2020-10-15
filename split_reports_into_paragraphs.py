# https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
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
from gensim.models import Doc2Vec, doc2vec
from sklearn import utils
import re

stopwords = set(stopwords.words('english'))

discharge_reports = open('data/discharge_reports_100samples.txt', 'r').read().split("\n|||\n")




# def process_data(fileName):
#     category = fileName.replace('.txt', '').replace('data/', '')
#     data = open(fileName, 'r').read().split("\n|||\n")
#     data_original = []
#     data_tokenized = []
#     for text in data:
#         text = text.lower()
#         text = re.sub("\[.*?\]", "", text)
#         text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
#         text_tokens_ns = [word for word in text_tokens if not word in stopwords]
#         data_original.append((' '.join(text_tokens_ns), category))
#         data_tokenized.append((text_tokens_ns, category))
#     return data, data_original, data_tokenized
