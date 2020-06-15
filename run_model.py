import pandas as pd
import gensim.corpora as corpora
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import json
from preprocessing import clean_tweets
from models import lda, biterm
from util import util

HASHTAGS = {r'#nlwhiteout', r'#nlweather', r'#newfoundland', r'#nlblizzard2020', r'#nlstm2020', r'#snowmaggedon2020', r'#stmageddon2020', r'#stormageddon2020', r'#newfoundland',
                            r'#snowpocalypse2020', r'#snowmageddon', r'#nlstm', r'#nlwx', r'#nlblizzard', r'#nlwx', 'snowmaggedon2020', 'newfoundland', r'#nlstorm2020'}

if __name__ == '__main__':
    num_topics = [5,10,15,20]

    winter_tweets = pd.read_csv('/home/smacawi/smacawi/repositories/tweet-classifier/nlwx_2020_hashtags_no_rt.csv')
    winter_tweets_cleaned = clean_tweets.preprocess(df=winter_tweets, extra_stopwords = HASHTAGS)
    ready_data = winter_tweets_cleaned['tokenized_text'].values.tolist()

    # get best LDA model
    #lda.get_best(ready_data, 1, 20, 1)

    best_num = 7
    lda_model = lda.build_model(ready_data, num_topics = best_num, include_vis = True)
    # save top words for each topic to csv
    lda.top_vocab(lda_model, 10)

    #for val in num_topics:
        #print("Building model with {} topics...".format(val))
    #     #lda_model = lda.build_model(ready_data, num_topics = val)
        #biterm_model = biterm.build_model(ready_data, num_topics = val)
    #




