from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
import pandas as pd
import re
import math

def get_frequent_vocab(corpus, threshold=0):
    '''
    Gets words whose frequency exceeds that of the threshold
    in a given corpus.

    :param corpus: list of tokenized words
    :param threshold:
    :return: list of words with higher frequency than threshold
    '''
    vect = CountVectorizer().fit(corpus)
    bag_of_words = vect.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    freq = {word: sum_words[0, idx] for word, idx in vect.vocabulary_.items()}
    return [word for word, count in freq.items() if count >= threshold]
