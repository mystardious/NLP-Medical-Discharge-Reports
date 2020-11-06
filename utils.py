import re

import matplotlib.pyplot as plt
import pandas as pd
from nltk.corpus import stopwords

import dataread


def write_string_to_file(string, filename):
    with open('data/' + filename, 'w') as f:
        f.write("%s" % string)


def write_array_to_file(data, filename):
    with open('data/' + filename, 'w') as f:
        for item in data:
            f.write("\"%s\",\n|||\n" % item)


def write_array_to_file(data, filename, pattern):
    with open('data/' + filename + ".txt", 'w') as f:
        for item in data:
            f.write("\"%s\",\n" % item)


def split_string(string, pattern):
    return string.split(pattern)


def print_array(array):
    for item in array:
        print(item)


def print_array(array, divider):
    for item in array:
        print(item + divider)


def remove_empty_array_items(array):
    ret_val = []
    for item in array:
        if ((len(item) > 0) and (not item.isspace())):
            ret_val.append(item)
    return ret_val


def clean_header(string):

    ret_val = string.lower()

    pattern_white_spaces_start_of_line = '^\s*(.+)'
    ret_val = re.findall(pattern_white_spaces_start_of_line, ret_val)

    return ret_val[0]


def find_all_section_headers(string):

    pattern = "\n([A-Za-z ]+):"
    stop_words = set(stopwords.words('english'))
    user_defined_stop_words = dataread.read_ignore_words(
        "section_header_count_ignore.txt")
    print(user_defined_stop_words)
    matches = re.findall(pattern, string)

    clean_matches = []
    for header in matches:
        clean_string = clean_header(header)
        if clean_string not in stop_words:
            if clean_string not in user_defined_stop_words:
                clean_matches.append(clean_string)
    return clean_matches


def count_all_section_headers(array):
    header_dict = {}

    for string in array:
        list_of_section_headers = find_all_section_headers(string)
        for key in list_of_section_headers:
            if key in header_dict:
                header_dict[key] = header_dict[key] + 1
            else:
                header_dict[key] = 1

    return header_dict


def count_all_section_headers_with_min_value(array, min_value):
    header_dict = count_all_section_headers(array)
    ret_val = {}
    for (key, value) in header_dict.items():
        if value > min_value:
            print("\"" + key + "\": "+str(value))
            ret_val[key] = value
    return ret_val


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
