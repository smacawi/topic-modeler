import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

def build_model(data, num_topics, include_vis = True):
    # Create Dictionary
    id2word = corpora.Dictionary(data)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data]

    # Build LDA model
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=num_topics,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=10,
                                                passes=10,
                                                alpha='symmetric',
                                                iterations=100,
                                                per_word_topics=True)

    #pyLDAvis
    if include_vis:
        p = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
        pyLDAvis.save_html(p, 'lda_{}.html'.format(num_topics))

    return lda_model

