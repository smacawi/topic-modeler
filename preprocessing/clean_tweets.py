from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from datetime import datetime
import nltk
nltk.download('stopwords')
import pandas as pd
import re
import math

def preprocess(df,
                 tweet_col='text',
                 to_lower = True,
                 include_bigram = False,
                 vocab_threshold = 10
                ):
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

    #extended stop words for twitter
    extended_stop_words = en_stop_words.union({
                              '&amp;', 'rt',
                              'th', 'co', 're', 've', 'kim', 'daca',
                            '#nlwhiteout', '#nlweather', 'newfoundland', '#nlblizzard2020', '#nlstm2020', '#snowmaggedon2020', '#stmageddon2020', '#stormageddon2020',
                            '#snowpocalypse2020', '#snowmageddon', '#nlstm', '#nlwx', '#nlblizzard'
                          })
    url_re = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    df_copy['preprocessed_' + tweet_col] = df_copy['preprocessed_' + tweet_col].apply(lambda row: ' '.join(
        [word for word in row.split() if (not word in extended_stop_words) and (not re.match(url_re, word))]))

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
        vocab = get_frequent_vocab(texts, threshold = vocab_threshold)
        df_copy['tokenized_' + tweet_col] = df_copy['tokenized_' + tweet_col].apply(lambda row: [word for word in row if word in vocab])

    #remove tweets with length less than two
    df_copy = df_copy[df_copy['tokenized_' + tweet_col].map(len) >= 2]

    return df_copy

def get_frequent_vocab(corpus, threshold=0):
    vect = CountVectorizer().fit(corpus)
    bag_of_words = vect.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    freq = {word: sum_words[0, idx] for word, idx in vect.vocabulary_.items()}
    return [word for word, count in freq.items() if count >= threshold]

#method that retrieves the most frequent words in the entire corpus
def get_most_freq_words(str, n=None):
    vect = CountVectorizer().fit(str)
    bag_of_words = vect.transform(str)
    sum_words = bag_of_words.sum(axis=0)
    freq = [(word, sum_words[0, idx]) for word, idx in vect.vocabulary_.items()]
    freq = sorted(freq, key=lambda x: x[1], reverse=True)
    return freq[:n]