import re
import pickle
import numpy as np
from konlpy.tag import Okt
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import KeyedVectors
import random
import time
from multiprocessing import Pool, freeze_support

device = torch.device('cuda')

class DataLoad():

    def __init__(self):
        self.word2idx, self.idx2freq, self.idx2word = self.get_unigram_freq()
        self.cooccur = pickle.load(open('../data/wiki/wiki_entity_cooccur_small.pickle', 'rb'))
        self.ent2idx = pickle.load(open('../data/wiki/wiki_ent2idx_small.pickle', 'rb'))
        self.word_embedding = KeyedVectors.load("../data/wiki/w2v/w2v_wiki_300.bin")
        self.unk = np.zeros(300, dtype=np.float32)

    def get_unigram_freq(self):
        word2idx = pickle.load(open('../data/wiki/wiki_word2idx_small.pickle', 'rb'))
        idx2freq = pickle.load(open('../data/wiki/wiki_idx2freq_small.pickle', 'rb'))

        idx2word = {}
        for key, value in word2idx.items():
            idx2word[value] = key
        
        return word2idx, idx2freq, idx2word

    def get_unigram_dist(self, freq_list): 
        dist = torch.tensor([f for i, f in freq_list], dtype=torch.float)
        dist = torch.pow(F.normalize(dist, p=1, dim=0), 0.6)

        return dist

    def get_positive_words(self, entity):
        entity_doc = self.cooccur[entity]
        idx2freq = {}
        idx2word = {}
        for i, (key, value) in enumerate(entity_doc.items()):
            idx2freq[i] = value
            idx2word[i] = key

        freq_list = sorted(idx2freq.items(), key=lambda x: x[0], reverse=False)
        dist = torch.tensor([f for i, f in freq_list], dtype=torch.float)
        dist = F.normalize(dist, p=1, dim=0)

        num = 20
        positive = torch.multinomial(dist, num, replacement=True)
        positive_words = [idx2word[freq_list[p.item()][0]] for p in positive]
        expects = [dist[p.item()].item() for p in positive]

        return positive_words, expects

    def get_negative_words(self, entity):
        entity_doc = self.cooccur[entity]
        sample = self.idx2freq.copy()
        for key in entity_doc.keys():
            idx = self.word2idx[key]
            del sample[idx]
        freq_list = random.sample(list(sample.items()), 10000)
        dist = self.get_unigram_dist(freq_list)
 
        negative = torch.multinomial(dist, 5).tolist()
        negative_words = [self.idx2word[freq_list[n][0]] for n in negative]
        expects = [dist[n].item() for n in negative]

        return negative_words, expects

    def get_next_batch(self, target):
        expects_list = []
        margins_list = []
        for t in target:  
            positive, pos_exp = self.get_positive_words(t[0])     
            negative, neg_exp = self.get_negative_words(t[0])
            expects = []
            # margins = []
            for p, p_e in zip(positive, pos_exp):
                try:
                    embed_p = self.word_embedding[p]
                except KeyError:
                    embed_p = self.unk
                for n, n_e in zip(negative, neg_exp):
                    try: 
                        embed_n = self.word_embedding[n]
                    except KeyError:
                        embed_n = self.unk
                    margin = embed_p - embed_n
                    margins_list.append(margin)
                    expect = p_e * n_e
                    expects.append(expect)
            expects_list += expects

        return margins_list, expects_list

if __name__ == "__main__":
    d = DataLoad()
    ent2idx = d.ent2idx.items()
    print(len(ent2idx))
    target = list(ent2idx)[:32]
    time1 = time.time()
    d.get_next_batch(target)
    time2 = time.time()
    print(time2 - time1)