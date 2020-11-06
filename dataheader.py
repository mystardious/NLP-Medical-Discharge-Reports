

import re

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords as sw

import dataread


def clean_string(string: str):
    """Clean section header string"""

    # Convert to lowercase
    ret_val = string.lower()

    # Remove whitespaces from beginning of line
    pattern_white_spaces_start_of_line = '^\s*(.+)'
    ret_val = re.findall(pattern_white_spaces_start_of_line, ret_val)

    return ret_val[0]


def find_header(string: str) -> list:
    """Find all possible Section Headers in a Sample"""

    stopwords = set(sw.words('english'))
    user_stopwords = dataread.read_header_ignore_words()
    pattern_header = "\n([A-Za-z ]+):"

    headers = re.findall(pattern_header, string)
    ret_val = []

    for header in headers:

        clean_header = clean_string(header)

        if (clean_header not in stopwords) and (clean_header not in user_stopwords):
            ret_val.append(clean_header)

    return ret_val


def count_header(array: list):

    ret_val = {}

    for sample in array:

        header_list = find_header(sample)
        for key in header_list:

            if key in ret_val:
                ret_val[key] = ret_val[key] + 1
            else:
                ret_val[key] = 1

    return ret_val


def count_header_min_floor(array: list, floor: int):

    header_list = count_header(array)
    ret_val = {}

    for (key, value) in header_list.items():

        if value > floor:
            ret_val[key] = value

    return ret_val


def plot_header_count(dictionary: dict, limit: int):
    """Show graph of highest count of headers"""

    series = pd.Series
    series = pd.Series(dictionary).sort_values(ascending=True)

    if(limit > 0):
        series = series.tail(limit)

    series.plot(kind='barh')
    plt.show()

