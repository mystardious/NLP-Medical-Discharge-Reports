import re

import pandas as pd

import dataread

no_samples = 1000
section_list = dataread.read_file('top_sectionheaders_50000.txt')
sample_list = dataread.read_samples(no_samples)

# pattern = ':[ \n]*(.*?)\n(?:[A-Z ]+:|\n)'


def sample_extract_section(section_name: str):

    ret_val = []

    pattern = section_name + ':(.*?)\n(?:[a-zA-Z\- ]+:|[\n]{2,})'
    
    # (?i)history of present illness:[ \n]*(.*?)\n(?:[A-Za-z ]+:|\n)
    # Social History:[ \n]*(.*?)(?:[\n]{2, }[a-zA-Z ]+:|[\n]{3,})
    # Social History:[ \n]*(.*?)\n(?:[a-zA-Z ]+:|[\n]{2,})

    for sample in sample_list:

        matches = re.findall(pattern, sample, flags=re.IGNORECASE | re.DOTALL)

        if matches:
            ret_val.append(matches[0])

    dataread.save_array(ret_val, 'section/' + section_name.lower().replace(' ', '_') + str(no_samples) + '.txt')


if __name__ == "__main__":

    for section in section_list:
        print('Extracting '+ section + ".....", end="")
        sample_extract_section(section)
        print('OK')

    # sample_extract_section("major surgical or invasive procedure")