from PrepareChain import *
from tqdm import tqdm

def storeTweetstoDB():
    
    with open('tweets.txt', 'r', encoding='utf-8') as f:
        tweets = [line.strip() for line in f]

    print(len(tweets))

    chain = PrepareChain(tweets[0])
    triplet_freqs = chain.make_triplet_freqs()
    chain.save(triplet_freqs, True)

    for i in tqdm(tweets[1:]):
        chain = PrepareChain(i)
        triplet_freqs = chain.make_triplet_freqs()
        chain.save(triplet_freqs, False)

if __name__ == '__main__':
    storeTweetstoDB()
