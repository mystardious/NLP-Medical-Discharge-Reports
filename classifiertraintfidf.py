# Import required packages
from sklearn.metrics import accuracy_score, classification_report
from random import shuffle
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import dataread
import classifierutils
import logging
import random
import re

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import sklearn
from nltk.corpus import stopwords
from nltk.tokenize.regexp import regexp_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

# https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

# List of stopwords
all_stopwords = set(stopwords.words('english'))

# pos_data = open("data/section_juan/allergies.txt", "r").read().split("\n|||\n")
# pos_document = []
# pos_p = []
# for text in pos_data:
#     text = text.lower()
#     text = re.sub("\[.*?\]", "", text)
#     text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
#     text_tokens_ns = [word for word in text_tokens if not word in all_stopwords]
#     pos_p.append((' '.join(text_tokens_ns), 'allergies'))
#     pos_document.append((text_tokens_ns, 'allergies'))

# neg_data = open("data/section_juan/social_history.txt", "r").read().split("\n|||\n")
# neg_document = []
# neg_p = []
# for text in neg_data:
#     text = text.lower()
#     text = re.sub("\[.*?\]", "", text)
#     text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
#     text_tokens_ns = [word for word in text_tokens if not word in all_stopwords]
#     neg_p.append((' '.join(text_tokens_ns), 'social_history'))
#     neg_document.append((text_tokens_ns, 'social_history'))

# neu_data = open("data/section_juan/family_history.txt", "r").read().split("\n|||\n")
# neu_document = []
# neu_p = []
# for text in neu_data:
#     text = text.lower()
#     text = re.sub("\[.*?\]", "", text)
#     text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
#     text_tokens_ns = [word for word in text_tokens if not word in all_stopwords]
#     neu_p.append((' '.join(text_tokens_ns), 'family_history'))
#     neu_document.append((text_tokens_ns, 'family_history'))

# ext_data = open("data/section_juan/family_history.txt", "r").read().split("\n|||\n")
# ext_document = []
# ext_p = []
# for text in ext_data:
#     text = text.lower()
#     text = re.sub("\[.*?\]", "", text)
#     text_tokens = regexp_tokenize(text, r"[a-zA-z]+")
#     text_tokens_ns = [word for word in text_tokens if not word in all_stopwords]
#     ext_p.append((' '.join(text_tokens_ns), 'history_illness'))
#     ext_document.append((text_tokens_ns, 'history_illness'))

# df = pos_p + neg_p + ext_p + neu_p
# random.shuffle(df)
# df = pd.DataFrame(df, columns=['Data', 'Category'])

# print(df)

# X = df['Data']
# y = df['Category']

# def w2v_tokenize_text(text):
#     tokens = []
#     for sent in nltk.sent_tokenize(text, language='english'):
#         for word in nltk.word_tokenize(sent, language='english'):
#             if len(word) < 2:
#                 continue
#             tokens.append(word)
#     return tokens


"""Variables"""

# headers = ['allergies', 'family_history', 'history_illness', 'social_history']
# headers = dataread.read_file('top_sectionheaders_50000.txt')
headers = dataread.read_file('custom.txt')
no_sections = 5000

"""Import data"""

# header --> [header, original, tokenized, tokenized_labelled]
header_corpus = {}

for header in headers:
    header_corpus[header] = {}
    header_corpus[header]['label'] = header
    temp = dataread.read_file('section/'+header.replace(' ',
                                                        '_')+str(no_sections)+'.txt')
    new = []
    for sample in temp:
        kek = re.sub("\[.*?\]", "", sample)
        new.append(kek)

    header_corpus[header]['original'] = pd.Series(
        new
    )
    temp = classifierutils.corpus_preprocess(
        header_corpus[header]['original'],
        header
    )
    header_corpus[header]['labelled'] = temp[0]
    header_corpus[header]['labelled_tokenised'] = temp[1]

mixed_labelled = pd.DataFrame()
for value in header_corpus.values():
    mixed_labelled = mixed_labelled.append(value['labelled'])

mixed_labelled_tokenised = pd.DataFrame()
for value in header_corpus.values():
    mixed_labelled_tokenised = mixed_labelled_tokenised.append(
        value['labelled_tokenised'])

mixed_labelled = shuffle(mixed_labelled)

print(mixed_labelled)

X = mixed_labelled['TEXT']
y = mixed_labelled_tokenised['HEADER']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# test_tokenized = X_test.apply(lambda r: w2v_tokenize_text(r[0])).values
# train_tokenized = X_train.apply(lambda r: w2v_tokenize_text(r[0])).values


nb = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', LogisticRegression(n_jobs=1, C=1e5)),
               ])
nb.fit(X_train, y_train)


y_pred = nb.predict(X_test)



print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))