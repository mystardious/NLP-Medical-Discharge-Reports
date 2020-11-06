import csv
import json
import pickle
import re
from typing import *

import pandas as pd
from gensim.models.doc2vec import Doc2Vec

filename = r'data/NOTEEVENTS.csv.gz'
divider = "\n|||\n"


def save_samples(no_samples: int):
    """Saves 'no_samples' amount of reports from the NOTEEVENTS table

    Parameters
    ----------
    no_samples : int
        The number of samples you want to save
    """
    read_file = pd.read_csv(filename, nrows=no_samples,
                            compression='gzip', error_bad_lines=False)
    with open('data/discharge_reports_' + str(no_samples) + 'samples.txt', 'w') as f:
        for item in read_file.TEXT:
            f.write(item + divider)


def save_array(array: List[str], filename: str):
    with open('data/' + filename, 'w') as f:
        for item in array:
            f.write(item + divider)


def save_dict(dictionary: dict, filename: str):
    """Keys are sorted Alphabetically"""
    with open('data/' + filename + '.json', 'w') as f:
        f.write(json.dumps(
            dict(sorted(dictionary.items()))
        ))


def save_classifier(model, algorithm: str = '', no_classes: str = '', feature_model: str = '', filename: str = ''):
    """Store trained model into a file

    Parameters
    ----------
    model
        The model to be saved
    algorithm : str
        Algorithm the classifier makes use of
    no_classes : str
        Number of classes the classifer will distinguish between
    feature_model : str
        The model used to word features
    filename : str, optional
        Custom filename specified by the user, by default a combination of the
        paramters listed are used
    """
    if not (filename):
        filename = algorithm + '_' + \
            str(no_classes) + 'classes' + '_' + feature_model

    pickle.dump(model, open('model/'+filename+'.sav', 'wb'))


def save_word_model(model: Doc2Vec, no_classes: str = '', filename: str = ''):

    if not (filename):
        filename = 'Doc2Vec' + '_' + str(no_classes) + "classes"

    model.save('model/' + filename + '.bin')


def read_file(filename: str, separator: str = '\n|||\n') -> List[str]:
    """Read txt file from data/'filename'"""
    with open('data/'+filename, 'r') as f:
        return f.read().split(separator)


def read_samples(no_samples: int = 0) -> List[str]:
    """Reads 'no_samples' amount of reports from the NOTEEVENTS table

    Parameters
    ----------
    no_samples : int
        The number of samples you want to read
    """
    if no_samples:
        read_file = pd.read_csv(
            filename, nrows=no_samples, compression='gzip', error_bad_lines=False)
    else:
        read_file = pd.read_csv(
            filename, compression='gzip', error_bad_lines=False)
    return read_file.TEXT


def read_json(filename: str) -> Dict:
    """Read json from data/'filename'.json"""
    with open('data/'+filename+'.json') as f:
        return json.load(f)


def read_header_ignore_words() -> List[str]:
    """List of Section Headers to ignore"""
    with open(r'data/section_header_count_ignore.txt') as f:
        return f.read().split("\n")


def read_classifier(filename: str):
    """Read saved classifier from file '.sav'"""
    return pickle.load(open('model/' + filename + '.sav', 'rb'))


def read_word_model(filename: str) -> Doc2Vec:
    """Read word embedding model"""
    return Doc2Vec.load('model/' + filename + '.bin')


if __name__ == "__main__":
    # print(read_samples(5))
    # print(read_json('section_header_count_greater_than_five')['abd'])
    read_header_ignore_words()
