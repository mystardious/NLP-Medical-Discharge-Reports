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


def process_data(fileName):
    category = fileName.replace('.txt', '').replace('data/', '')
    data = open(fileName, 'r').read().split("\n|||\n")
    data_original = []
    data_tokenized = []
    for text in data:
        text = text.lower()
        text = re.sub("\[.*?\]", "", text)
        text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
        text_tokens_ns = [word for word in text_tokens if not word in stopwords]
        data_original.append((' '.join(text_tokens_ns), category))
        data_tokenized.append((text_tokens_ns, category))
    return data, data_original, data_tokenized

allergies_data, allergies_original, allergies_tokenized = process_data('data/allergies.txt')
family_history_data, family_history_original, family_history_tokenized = process_data('data/family_history.txt')
history_illness_data, history_illness_original, history_illness_tokenized = process_data('data/history_illness.txt')
social_history_data, social_history_original, social_history_tokenized = process_data('data/social_history.txt')

# data - Just print the whole sentence
# original - Same as data but has a label applied based on the file name of the imported file
# tokenized - Same as original but words are split into tokens e.g. ['patient', 'known', 'drug', 'allergies']
print(allergies_tokenized)

mixed_original = allergies_original + family_history_original + history_illness_original + social_history_original
mixed_tokenized = allergies_tokenized + family_history_tokenized + history_illness_tokenized + social_history_tokenized

random.shuffle(mixed_original)
mixed_original = pd.DataFrame(mixed_original)
mixed_original.columns = ['Data', 'Category']
