import re
from random import shuffle

import cufflinks
import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from sklearn.utils import shuffle

import classifierutils
import dataread

STOPWORDS = set(stopwords.words('english'))

InteractiveShell.ast_node_interactivity = 'all'

cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')

# https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568

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

mixed_labelled.HEADER.value_counts()
mixed_labelled['HEADER'].value_counts().sort_values(ascending=False).iplot(kind='bar', yTitle='Number of Samples',
                                                                           title='Number of samples in each section header')

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 50000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
# This is fixed.
EMBEDDING_DIM = 100

tokenizer = Tokenizer(num_words=MAX_NB_WORDS,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(mixed_labelled['TEXT'].values)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


model = load_model('model/LTSM_RNN.h5')

test = ["""No Known Drug Allergies"""]
seq = tokenizer.texts_to_sequences(test)
padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
pred = model.predict(padded)
labels = sorted(headers)
print(labels)
print(pred, labels[np.argmax(pred)])
