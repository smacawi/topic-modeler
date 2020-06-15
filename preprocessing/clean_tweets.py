from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from datetime import datetime
import nltk
nltk.download('stopwords')
import pandas as pd
import re
import math
from util import util

def preprocess(df,
                 extra_stopwords,
                 tweet_col='text',
                 to_lower = True,
                 include_bigram = False,
                 vocab_threshold = 10
                ):
    '''
    Preprocesses text in a certain dataframe column.

    :param df: dataframe to be processed
    :param tweet_col: name of dataframe column containing text
    :param to_lower: flag whether to convert to lowercase
    :param include_bigram: flag whether to include bigrams
    :param vocab_threshold: minimum number of words in each tweet
    :param extra_stopwords: extra stopwords to add onto default nltk
    :return: original df with extra column, 'tokenized_[tweet_col]'
    '''
    df_copy = df.copy()

    # drop rows with empty values
    df_copy.dropna(how='all', inplace=True)
    # drop rows with identical text
    df_copy.drop_duplicates(subset = 'text', inplace = True)

    # lower the tweets
    if to_lower:
        df_copy['preprocessed_' + tweet_col] = df_copy[tweet_col].str.lower()
    else:
        df_copy['preprocessed_' + tweet_col] = df_copy[tweet_col].str

    # filter out stop words and URLs
    en_stop_words = set(stopwords.words('english'))

    #extended stop words for twitter and extra stop words from user
    extended_stop_words = en_stop_words.union({
                              '&amp;', 'rt',
                              'th', 'co', 're', 've', 'kim', 'daca'
                          }).union(extra_stopwords)

    #removes urls
    url_re = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    df_copy['preprocessed_' + tweet_col] = df_copy['preprocessed_' + tweet_col].apply(lambda row: ' '.join(
        [word for word in row.split() if (word not in extended_stop_words) and (not re.match(url_re, word)) and (word.find('snowmageddon2020') == -1)]))

    # tokenize the tweets
    tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
    df_copy['tokenized_' + tweet_col] = df_copy['preprocessed_' + tweet_col].apply(lambda row: tokenizer.tokenize(row))

    if include_bigram:
        # Build the bigram and trigram models
        texts = df_copy['tokenized_' + tweet_col].values.tolist()
        bigram = Phrases(texts, min_count=5, threshold=100)  # higher threshold fewer phrases.
        bigram_mod = Phraser(bigram)
        df_copy['tokenized_' + tweet_col] = pd.Series([bigram_mod[doc] for doc in texts])

    # remove terms with frequency less than 10
    if vocab_threshold > 0:
        texts = [word for tweet in df_copy.tokenized_text for word in tweet]
        vocab = util.get_frequent_vocab(texts, threshold = vocab_threshold)
        df_copy['tokenized_' + tweet_col] = df_copy['tokenized_' + tweet_col].apply(lambda row: [word for word in row if word in vocab])

    #remove tweets with length less than two
    df_copy = df_copy[df_copy['tokenized_' + tweet_col].map(len) >= 2]

    return df_copy