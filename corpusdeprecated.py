# import re
# from typing import *

# from pandas import *

# from dataheader import *


# class Sample:
#     """A class used to represent a list of Samples"""

#     def __init__(self, data: str):
#         self.data = data
#         self.data_paragraph = self.split_data_paragraph()
#         self.data_headers = self.get_section_header()

#     def split_data_paragraph(self) -> List[str]:
#         "Return a list of this Sample split into paragraphs"
#         pattern_split_paragraph = '\n\s*\n+'
#         paragraph_list = re.split(pattern_split_paragraph, self.data)
#         return paragraph_list

#     def get_section_header(self) -> List[str]:
#         """Return a list of all possible section headers for this Sample"""
#         return find_header(self.data)

#     def __repr__(self):
#         ret_val = "----------\n"
#         separator = "\n----------\n"
#         for paragraph in self.data_paragraph:
#             ret_val = ret_val + paragraph + separator
#         return ret_val

#     def __str__(self):
#         return "<Sample data:%s>" % self.data


# class Corpus:
#     """A class used to represent a list of Samples"""

#     def __init__(self):
#         self.data = []

#     def add_sample(self, sample: Sample) -> None:
#         """Add a Sample to the Corpus"""
#         if(len(sample.data) == 0):
#             print("Warning! Sample does not contain any data")
#         else:
#             self.data.append(sample)

#     def add_string(self, sample: str) -> None:
#         """Create a Sample object and add it to the corpus"""
#         if(len(sample) == 0):
#             print("Warning! Sample does not contain any data")
#         else:
#             self.data.append(Sample(sample))

#     def count_section_header(self, floor: int = 0) -> Dict[str, int]:
#         """Return a count of the number of section headers"""
#         ret_val = {}

#         # Count all possible section headers
#         for sample in self.data:

#             header_list = find_header(sample.data)
#             for key in header_list:

#                 if key in ret_val:
#                     ret_val[key] = ret_val[key] + 1
#                 else:
#                     ret_val[key] = 1

#         # Filter count by specifying a minimum count value
#         if floor:

#             header_list = ret_val
#             ret_val = {}

#             for (key, value) in header_list.items():

#                 if value > floor:
#                     ret_val[key] = value

#         return ret_val
