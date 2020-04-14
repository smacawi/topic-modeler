from sklearn.feature_extraction.text import CountVectorizer
from biterm.utility import vec_to_biterms, topic_summuary
from biterm.btm import oBTM
import numpy as np
import pyLDAvis
import json

def identity_tokenizer (tokens):
    return tokens

def build_model(data, num_topics, include_vis= False):
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

    return biterm_model

