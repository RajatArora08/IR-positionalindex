"""
Author: Rajat Arora

This program accepts queries from user (normal and using proximity operator),
and finds the documents that contain the queried words, using the 'dictionary.txt'
file as positional inverted index.
"""

import re
from nltk.stem.porter import PorterStemmer
import ast
from math import log10

DIR = ""

TOTAL_DOCS = 10
DATA_FILE = DIR + "documents.txt"
DICTIONARY_FILE = DIR + "dictionary.txt"
RESULT_FILE = DIR + "results.txt"
final_dictionary = {}
final_doc_set = set()


def fetch_dictionary(query):
    """
    This function reads the dictionary file, and extracts the posting list
    for all the terms in the query. Pre-processing is done where ever required.
    """

    dictionary = {}

    query = re.sub(r'[0-9]*\(', '', query)
    query = re.sub(r'\)', '', query)
    query_words = [pre_process(term) for term in query.split()] # Terms are pre-processed before fetched

    pattern_word = re.compile('^[a-z0-9]*')
    pattern_list = re.compile('^[a-z0-9]*: ')

    with open(DICTIONARY_FILE, "r") as file:
        for line in file:
            word_in_dictionary = re.findall(pattern_word, line)[0]

            if any(word_in_dictionary == word for word in query_words):
                result = re.split(pattern_list, line)

                # ast library allows the text to be converted directly to dictionary to appropriate data type
                dictionary[word_in_dictionary] = ast.literal_eval(result[1])

    file.close()

    return dictionary


def pre_process(word):
    """
    This function removes symbols from each input word and stems it
    using the PorterStemmer of NLTK library. This function is called where ever required.
    """

    stemmer = PorterStemmer()
    return stemmer.stem(re.sub(r'[^\w\s]', '', word))


def fetch_posting_list(query):
    """
    This function creates a set of document ids for all qualified documents.
    This function ensures that the proximity operator are dealt with if present
    """

    global final_dictionary
    global final_doc_set

    proximity_window_pattern = re.compile('^[0-9]*')
    proximity_operator_pattern = re.compile('[0-9]*\([0-9a-z ]*\)')

    proximity_operator_list = re.findall(proximity_operator_pattern, query)

    if proximity_operator_list:
        for item in proximity_operator_list:
            proximity_query = item[item.find("(") + 1: item.find(")")]
            query = query.replace(item, '')
            # The proximity operator is processed in below function
            process_proximity_operator(int(re.search(proximity_window_pattern, item).group()),
                                       proximity_query.split())

    # All the query terms other than proximity operator are processed below.
    # All the documents ids are added to the the global set since it is OR relation.
    query_words = query.split()
    if query_words:
        for word in query_words:
            for key in final_dictionary[pre_process(word)][1]:
                final_doc_set.add(key)

    return


def validate_proximity(proximity_window, first_doc, second_doc):
    """
    This function accepts the posting list of two operands of the proximity query.
    It validates the indices of the terms in the documents against the input proximity window.
    """

    for index_first in first_doc[1]:
        for index_second in second_doc[1]:
            proximity = index_second-index_first
            if abs(proximity)-1 <= proximity_window and index_second > index_first:
                return True

    return False


def process_proximity_operator(proximity_window, query_words):
    """
    This function processes all the query portions that contain the proximity operator.
    The function adds all the qualified documents ids into to the global document set.
    """

    global final_dictionary
    global final_doc_set

    posting_list_first_term = final_dictionary[pre_process(query_words[0])][1]
    posting_list_second_term = final_dictionary[pre_process(query_words[1])][1]

    for key_first, value_first in posting_list_first_term.items():
        for key_second, value_second in posting_list_second_term.items():

            if key_first == key_second and \
                    validate_proximity(proximity_window, value_first, value_second):
                final_doc_set.add(key_first)

            elif key_first < key_second:
                break

    return


def tf_idf_score():
    """
    This function read the global document set and returns a scored list of document ids,
    using tf.idf scoring.
    """

    global final_doc_set
    global final_dictionary
    final_score = []

    for doc_id in final_doc_set:
        score = 0
        for query_term in final_dictionary.keys():
            if final_dictionary[query_term][1].get(doc_id):
                tf = final_dictionary[query_term][1][doc_id][0]
                df = final_dictionary[query_term][0]

                score += ((1 + log10(tf)) * log10(TOTAL_DOCS / df))

        final_score.append([doc_id, score])

    return final_score


def fetch_document_contents(query, doc_list):
    """
    This function reads and fetched the contents of all the doc id's
    present in input doc_list
    """

    output_str = 'Query [{0}] fetched {1} results:\n'.format(query, len(doc_list))
    flag = False
    doc_list.sort(key=lambda x: x[1], reverse=True)
    contents = {}

    with open(DATA_FILE, "r") as file:
        for line in file:
            if re.match("<DOC .*", line):  # New Document starts
                doc_id = int(re.search(r'\d+', line).group())
                contents[doc_id] = ''
                flag = True

            if flag:
                contents[doc_id] += line
                if re.match("</DOC.*", line):
                    flag = False

    file.close()

    for item in doc_list:
        output_str += '--------------------------------------------------------------\n'
        output_str += 'Document= {0} (Score= {1})\n'.format(item[0], item[1])
        output_str += contents[item[0]]

    return output_str


def main():
    """
    This function is the main entry point for the program.
    It accepts a query from user as input and searched it.
    """

    global final_dictionary
    global final_doc_set

    input_query = input("Please enter query for search: ")

    # Retrieving positional inverted index for query terms
    final_dictionary = fetch_dictionary(input_query.lower()) # Query is converted to lowercase as pre-process step

    #The final set of document IDs is retrieved below
    fetch_posting_list(input_query)
    sc = tf_idf_score()
    output = fetch_document_contents(input_query, sc)
    print(output)
    output_file = open(RESULT_FILE, 'a')
    output_file.write(output)
    output_file.write('\n##############################################################\n')
    output_file.close()

    print("Query results also appended to file: {0}".format(RESULT_FILE))


main()
