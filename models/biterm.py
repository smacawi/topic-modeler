from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary
from biterm.btm import oBTM
import numpy as np
import pandas as pd
import pyLDAvis
import json

def identity_tokenizer (tokens):
    return tokens

def build_model(data, num_topics, include_vis= False):
    '''
    Builds biterm model based on parameters,
    saves output model in a npy model
    and outputs pyLDAvis model

    :param data: list of preprocessed text
    :param num_topics: number of topics for biterm model
    :param include_vis: flag to include
    :return: biterm model
    '''

    texts = [' '.join(tweet) for tweet in data]

    #vectorize tokens
    vec = CountVectorizer(stop_words='english', lowercase=False)
    X = vec.fit_transform(texts).toarray()

    #build vocabulary
    vocab = np.array(vec.get_feature_names())
    biterms = vec_to_biterms(X)

    btm = oBTM(num_topics=num_topics, V=vocab)
    biterm_model = btm.fit_transform(biterms, iterations=100)

    for i in range(len(texts[:50])):
        print("{} (topic: {})".format(texts[i], biterm_model[i].argmax()))

    res = topic_summuary(btm.phi_wz.T, X, vocab, 10)
    np.save('btm_summary_{}.npy'.format(num_topics), res)

    #pyLDAvis
    if include_vis:
        vis = pyLDAvis.prepare(btm.phi_wz.T, biterm_model, np.count_nonzero(X, axis=1), vocab, np.sum(X, axis=0))
        pyLDAvis.save_html(p, 'biterm_{}.html'.format(num_topics))

    return btm, biterm_model

def top_vocab (model_file):
    '''
    Saves top 10 words per topic given model .npy file

    :param model_file: file location of npy biterm model
    :return: saves csv of top words per topic
    '''
    btm = np.load(model_file, allow_pickle=True)
    df = pd.DataFrame()

    #save top words per topic
    for i in range(len(btm.item()['top_words'])):
        df['topic_{}'.format(i)] = pd.Series(btm.item()['top_words'][i])
    df.to_csv('btm_top_words_{}.csv'.format(len(btm.item()['top_words'])))
    return df.T.values.tolist()

def get_best(data, lo, hi, step):
    '''
    Trains biterm for varied number of topics and provides coherence information for each to optimize
    for number of topics.

    :param data: list of tweets to be processed
    :param lo: lower bound of topic number to consider
    :param hi: upper bound of topic number to consider
    :param step: step size for topic number iteration
    :return: saves plot of coherence scores for each topic model and csv of coherence scores
    '''
    tweets_coherence = []
    for nb_topics in range(lo, hi, step):
        btm = build_model(data, nb_topics)
        tweets_coherence.append(sum(btm.item()['coherence']) / len(btm.item()['coherence']))

    print(tweets_coherence)
    df = pd.DataFrame(data={"num_topics": range(lo, hi, step), "coherence": tweets_coherence})
    df.to_csv("btm_coherence.csv", sep=',', index=False)



