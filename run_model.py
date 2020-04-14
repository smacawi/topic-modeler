import pandas as pd
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import json
from preprocessing import clean_tweets
from models import lda, biterm

def get_best_lda(data, lo, hi):
    tweets_coherence = []
    # Create Dictionary
    id2word = corpora.Dictionary(data)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data]
    for nb_topics in range(lo, hi):
        lda_model = lda.build_model(data, num_topics = nb_topics, include_vis = False)
        cohm = CoherenceModel(model=lda_model, corpus=corpus, dictionary=id2word, coherence='u_mass')
        coh = cohm.get_coherence()
        tweets_coherence.append(coh)

    # visualize coherence
    plt.figure(figsize=(10, 5))
    plt.plot(range(lo, hi), tweets_coherence)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score");
    plt.savefig('lda_coherence.png', bbox_inches='tight')

if __name__ == '__main__':
    num_topics = [5,10,15,20]

    winter_tweets = pd.read_csv('/home/smacawi/smacawi/repositories/tweet-classifier/nlwx_2020_hashtags_no_rt.csv')
    winter_tweets_cleaned = clean_tweets.preprocess(winter_tweets)

    #save df with preprocessed text and tokenized text
    #winter_tweets_cleaned.to_csv('./nlwx_2020_hashtags_no_rt_cleaned.csv')

    #most_freq_words = clean_tweets.get_most_freq_words([word for tweet in winter_tweets_cleaned.tokenized_text for word in tweet], 10)

    ready_data = winter_tweets_cleaned['tokenized_text'].values.tolist()

    #optimize number of topics
    # get_best_lda(ready_data, 1, 20)

    # best_num = 7
    # lda_model = lda.build_model(ready_data, num_topics = best_num, include_vis = True)
    # # save top words for each topic to csv
    # top_words_per_topic = []
    # for t in range(lda_model.num_topics):
    #     top_words_per_topic.extend([(t,) + x for x in lda_model.show_topic(t, topn=10)])
    # pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_words_{}.csv".format(best_num))

    for val in num_topics:
        print("Building model with {} topics...".format(val))
    #     #lda_model = lda.build_model(ready_data, num_topics = val)
        biterm_model = biterm.build_model(ready_data, num_topics = val)
    #
    #     # #save top words for each topic to csv
    #     # top_words_per_topic = []
    #     # for t in range(lda_model.num_topics):
    #     #     top_words_per_topic.extend([(t,) + x for x in lda_model.show_topic(t, topn=10)])
    #     # pd.DataFrame(top_words_per_topic, columns=['Topic', 'Word', 'P']).to_csv("top_words_{}.csv".format(val))




