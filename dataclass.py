import re

import chart_studio.plotly as py
import cufflinks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objs as go
import seaborn as sns
from bs4 import BeautifulSoup
from IPython.core.interactiveshell import InteractiveShell
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical
from nltk import word_tokenize
from nltk.corpus import stopwords
from plotly.offline import iplot
from sklearn.model_selection import train_test_split

import dataread

STOPWORDS = set(stopwords.words('english'))

InteractiveShell.ast_node_interactivity = 'all'

cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


STOPWORDS = set(stopwords.words('english'))

InteractiveShell.ast_node_interactivity = 'all'

cufflinks.go_offline()
cufflinks.set_config_file(world_readable=True, theme='pearl')


class Sample:

    def __init__(self, data):
        self.original = data
        self.paragraph = re.split('\n\s*\n', data)

    def paragraph_classify(self, model, tokenizer, headers):

        paragraph_classified = []

        # The maximum number of words to be used. (most frequent)
        MAX_NB_WORDS = 50000
        # Max number of words in each complaint.
        MAX_SEQUENCE_LENGTH = 250
        # This is fixed.
        EMBEDDING_DIM = 100

        for text in self.paragraph:

            seq = tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH)
            pred = model.predict(padded)

            paragraph_classified.append((
                headers[np.argmax(pred)],
                text,
                pred
            ))

        self.paragraph_classifed = paragraph_classified

    def paragraph_classify_indexed(self):

        ret_val = []
        original_lines = self.original.splitlines()

        for i in range(len(self.paragraph_classifed)):

            index_start = -1
            index_finish = -1
            text_split = self.paragraph_classifed[i][1].splitlines()

            if(len(text_split) > 0):
                text_first_line = text_split[0]
                text_last_line = text_split[-1]

                for index_line in range(len(original_lines)):
                    if text_first_line in original_lines[index_line]:
                        index_start = index_line+1
                    if text_last_line in original_lines[index_line]:
                        index_finish = index_line+1

                ret_val.append(
                    (*self.paragraph_classifed[i], (index_start, index_finish)))

        self.paragraph_classifed_indexed = ret_val

    def paragraph_classifed_print(self):

        ret_val = "Classification Report for Sample\n"
        ret_val += self.original
        for item in self.paragraph_classifed_indexed:
            print(sum(item[2][0]), len(item[2][0]))
            ret_val += item[0] + ": " + \
                str(item[3][0]) + "-" + str(item[3][-1]) + '\n'
        return ret_val


if __name__ == "__main__":
    x = Sample(dataread.read_samples(5)[3])
    x.paragraph_classify(x)
    x.paragraph_classify_indexed()
    print(len(x.paragraph_classifed_indexed))
    print(x.paragraph_classifed_indexed)
    print(x.paragraph_classifed_print())
    # print(x.paragraph_classify_indexed())
    # print(x.)
