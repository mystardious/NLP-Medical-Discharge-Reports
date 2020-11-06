# https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
import collections
import json
import random
import re

import numpy as np
import pandas as pd
# from gensim.models import Doc2Vec, doc2vec
from nltk.corpus import stopwords
from nltk.tokenize.regexp import regexp_tokenize
from sklearn import utils
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import dataread
import utils

pattern_empty_newline = "\n\n"


def removeEmptyLines(array):
    ret_val = []
    for item in array:
        temp = utils.split_string(item, pattern_empty_newline)
        temp = utils.remove_empty_array_items(temp)
        ret_val.append(temp)
    return ret_val


stopwords = set(stopwords.words('english'))

discharge_reports = open(
    'data/discharge_reports_100samples.txt', 'r').read().split("\n|||\n")

df = pd.DataFrame(data=discharge_reports, columns=['text'])

dataread.save_dict(
    utils.count_all_section_headers_with_min_value(df['text'], 5),
    'test'
)

# utils.write_string_to_file(
#     json.dumps(dict(
#         sorted(utils.count_all_section_headers_with_min_value(df['text'], 5).items()))),
#     "100samples_section_header_count_greater_than_five.json"
# )


# utils.print_array(
#     utils.remove_empty_array_items(utils.split_string(
#         df['text'].iloc[0],
#         pattern_empty_newline)), "\n|||")

# print("Without Spaces! \n\n")
# for item in removeEmptyLines(df['text'])[0]:
#     print(item)
#     print("|||")


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
