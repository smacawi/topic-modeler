import numpy as np
import pandas as pd

def get_biterm(num_topics):
    btm = np.load('btm_summary_{}.npy'.format(num_topics), allow_pickle=True)
    df = pd.DataFrame()

    #save top words per topic
    #for i in range(len(btm.item()['top_words'])):
        #df['topic_{}'.format(i)] = pd.Series(btm.item()['top_words'][i])
    #df.to_csv('btm_top_words_{}.csv'.format(num_topics))

if __name__ == "__main__":
    topics = [5,10,15,20]

    tweets_coherence = []
    for num in topics:
        btm = np.load('btm_summary_{}.npy'.format(num), allow_pickle=True)
        print(len(btm.item()['coherence']))
        tweets_coherence.append(sum(btm.item()['coherence']) / len(btm.item()['coherence']))
        #get_biterm(num)
    print(tweets_coherence)
    df = pd.DataFrame(data={"num_topics": range(5, 21, 5), "coherence": tweets_coherence})
    df.to_csv("btm_coherence.csv", sep=',', index=False)