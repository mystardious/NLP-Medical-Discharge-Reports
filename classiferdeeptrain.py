"""Main file used to train word embeddings and classifier

   Makes use of a range of helper methods from the following files:

    - classifierutils.py: Which processes the data in a format that is easy to
      feed for training.
    - dataread.py: Import and prepare data from data folder e.g. NOTEEVENTS table
    - dataheader.py: Extract section header information from medical reports.
    - datasection.py: Export sections from medical reports and save to data folder

"""

import random
import re

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec, doc2vec
from nltk.corpus import stopwords
from nltk.tokenize.regexp import regexp_tokenize
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

import classifierutils
import dataread

tqdm.pandas(desc="progress-bar")

"""Variables"""

# headers = ['allergies', 'family_history', 'history_illness', 'social_history']
headers = dataread.read_file('top_sectionheaders_50000.txt')
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

"""Train Vocab"""


def label_sentences(corpus, label_type):
    """
    Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
    We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
    a dummy index of the post.
    """
    labeled = []
    for i, v in enumerate(corpus):
        label = label_type + '_' + str(i)
        labeled.append(doc2vec.TaggedDocument(v.split(), [label]))
    return labeled


no_samples = 10000
vec_size = 300
X_train, X_test, y_train, y_test = train_test_split(mixed_labelled.head(no_samples).TEXT, mixed_labelled.head(no_samples).HEADER, random_state=0,
                                                    test_size=0.5)
X_train = label_sentences(X_train, 'Train')
X_test = label_sentences(X_test, 'Test')
all_data = X_train + X_test

model_dbow = Doc2Vec(dm=0, vector_size=vec_size, negative=5,
                     min_count=1, alpha=0.065, min_alpha=0.065, workers=4)
model_dbow.build_vocab([x for x in tqdm(all_data)])

for epoch in range(30):
    model_dbow.train([x for x in tqdm(all_data)],
                     total_examples=len(all_data), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

# docvec = doc2vec.vec

# max_epochs = 2
# vec_size = 100
# alpha = 0.025

# model_dbow = Doc2Vec(vec_size=vec_size,
#                 alpha=alpha,
#                 min_alpha=0.00025,
#                 min_count=1,
#                 dm =1,
#                 workers = 8)

# model_dbow.build_vocab([x for x in tqdm(all_data)])

# for epoch in range(max_epochs):
#     print('iteration {0}'.format(epoch))
#     model_dbow.train([x for x in tqdm(all_data)], total_examples=model_dbow.corpus_count, epochs=model_dbow.epochs)
#     # decrease the learning rate
#     model_dbow.alpha -= 0.0002
#     # fix the learning rate, no decay
#     model_dbow.min_alpha = model_dbow.alpha

# max_epochs = 30
# model_dbow = Doc2Vec()

# model_dbow.build_vocab([x for x in tqdm(all_data)])
# model_dbow.train([x for x in tqdm(all_data)], total_examples=len(all_data), epochs=max_epochs)

# Save word embedding model

# model_dbow = Doc2Vec.load('mymodel.bin')


def get_vectors(model, corpus_size, vectors_size, vectors_type):
    """
    Get vectors from trained doc2vec model
    :param doc2vec_model: Trained Doc2Vec model
    :param corpus_size: Size of the data
    :param vectors_size: Size of the embedding vectors
    :param vectors_type: Training or Testing vectors
    :return: list of vectors
    """
    vectors = np.zeros((corpus_size, vectors_size))
    for i in range(0, corpus_size):
        prefix = vectors_type + '_' + str(i)
        vectors[i] = model.docvecs[prefix]
    return vectors


train_vectors_dbow = get_vectors(model_dbow, len(X_train), vec_size, 'Train')
test_vectors_dbow = get_vectors(model_dbow, len(X_test), vec_size, 'Test')

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(train_vectors_dbow, y_train)
logreg = logreg.fit(train_vectors_dbow, y_train)

# Save classifier model
dataread.save_classifier(logreg, 'LogisticRegression', 4, 'Doc2Vec')

y_pred = logreg.predict(test_vectors_dbow)
print('accuracy %s' % accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred))

# model_dbow.save('model/Doc2Vec_4classes')
