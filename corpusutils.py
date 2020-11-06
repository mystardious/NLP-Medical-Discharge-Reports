import re
from typing import *

from dataheader import *
from mytypes import *


import re
from typing import *

import corpusutils
import dataread
from mytypes import *


class Sample:
    """A class used to represent a list of Samples"""

    def __init__(self, sample: str, header_index: HeaderIndexDict = None):
        self.data = sample
        if header_index:
            self.header_index = header_index


class Corpus:
    """A class used to represent a list of Samples

    Parameters
    ----------
    no_samples : int, optional
        Number of samples to import, by default all samples are imported
    header_index : bool, optional
        Calculate header indexs, by default False
    """

    def __init__(self, no_samples: int = 0, header_index: bool = False):
        """Initiliase Corpus class

        Parameters
        ----------
        no_samples : int, optional
            Number of samples to import, by default all samples are imported
        header_index : bool, optional
            Calculate header indexs, by default False
        """
        self.data = []

        if no_samples:
            if header_index:
                sample_list = dataread.read_samples(no_samples)
                header_dictionary = dataread.read_json(
                    '1000samples_section_header_count_greater_than_five').keys()
                for sample in sample_list:
                    self.data.append(Sample(
                        sample,
                        corpusutils.sample_index_headers(
                            sample, header_dictionary)
                    ))
            else:
                self.data = dataread.read_samples(no_samples)

    def add_string(self, sample: str):
        self.data.append(sample)

    def add_sample(self, sample: Sample):
        self.data.append(sample)


def sample_split_chunks(sample: str, pattern: str = '\n\s*\n+') -> SampleChunk:
    """Return sample in chunks divided by specified pattern

    Parameters
    ----------
    sample : str
        A 'sample' to divide into chunks
    pattern : str, optional
        Specify pattern you want to split sample
    """
    return re.split(pattern, sample)


def sample_index_headers(sample: str, header_dictionary: HeaderCountDict = None) -> HeaderIndexDict:
    """Return dictionary indicating which lines a section header starts and ends

    Parameters
    ----------
    sample : str
        Sample to index Section Headers
    header_dictionary : HeaderCountDict, optional
        Dictionary to identify Section Headers, by default imported from file

    """
    


def sample_concat_chunks(array: SampleChunk, separator: str = "\n----------\n") -> str:
    """Return a string with the items in the array divided by a separator

    Parameters
    ----------
    array : SampleChunk
        Sample document divided into chunks
    """
    ret_val = separator
    for string in array:
        ret_val = ret_val + string + separator
    return ret_val
