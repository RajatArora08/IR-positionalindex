"""
Author: Rajat Arora

This program takes 'document.txt' file as input and generates
a positional inverted index. Output is written to file 'dictionary.txt'
"""

import re
from nltk.stem.porter import PorterStemmer
from collections import OrderedDict
import pickle

DATA_FILE = "documents.txt"
DICTIONARY_FILE = "dictionary.txt"
STOP_WORDS = ['the', 'is', 'at', 'of', 'and', 'a']  # Other stop words can be added here


def tokenize_normalize(line):
    """
    This function tokenize, normalizes and lowers the case of the
    line read from input file
    """

    return re.sub(r'[^\w\s]', '', line).lower()


def generate_positional_indices(word_token_stream):
    """
    This function accepts a token stream and generates the postional indices and returns them.
    Word pre-processing is also done in this step.
    """

    index = -1
    token_stream = {}
    stemmer = PorterStemmer()

    for word in word_token_stream:
        index += 1
        if word in STOP_WORDS:  # Stop word removal
            continue
        word = stemmer.stem(word)
        if word in token_stream:
            token_stream[word].append(index)
        else:
            token_stream[word] = [index]
    return token_stream


def generate_token_sequence(input_file):
    """
    This function converts accepts a file name and generates
    a processed token sequence of all the terms in the document
    along with their doc id's.
    """

    token_list_all = {}
    line_concat = ''

    with open(input_file, "r") as file:
        for line in file:

            if re.match("<DOC .*", line):  # New Document starts
                doc_id = int(re.search(r'\d+', line).group())

            elif re.match("</DOC.*", line):
                token_list_all[doc_id] = generate_positional_indices(line_concat.split())
                line_concat = ''
                continue

            else:
                normalised_line = tokenize_normalize(line)  # Tokenize, Normalize
                line_concat += normalised_line

    file.close()

    return token_list_all


def generate_dictionary(token_sequence, dictionary_file):
    """
    This functions accepts a token sequence of processed word read
    from the input file, generates a sorted dictionary of all words
    along with their posting list, and write to a dictionary file
    """

    word_dictionary = {}
    output_file = open(dictionary_file, "w")

    for doc_id in token_sequence.keys():
        for word in token_sequence[doc_id].keys():
            if word in word_dictionary:
                word_dictionary[word][0] += 1
                word_dictionary[word][1][doc_id] = [len(token_sequence[doc_id][word]), token_sequence[doc_id][word] ]
            else:
                word_dictionary[word] = [1, {doc_id : [len(token_sequence[doc_id][word]), token_sequence[doc_id][word] ] }]

    # The dictionary is sorted in ascending order before it is written to file
    for key, value in OrderedDict(sorted(word_dictionary.items(), key=lambda t: t[0])).items():
        output_file.write("{0}: {1}\n".format(key, value)) # Writing to file

    output_file.close()
    print("Dictionary written to file: %s" % dictionary_file)

    return OrderedDict(sorted(word_dictionary.items(), key=lambda t: t[0]))


# Generating processed token sequence of words
token_sequence = generate_token_sequence(DATA_FILE)

# Generating the dictionary and writing to file
final_word_dictionary = generate_dictionary(token_sequence, DICTIONARY_FILE)
