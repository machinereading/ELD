import re
import pickle
import numpy as np
from konlpy.tag import Okt
from collections import Counter
from wikidict import basic_prob
import argparse

parser = argparse.ArgumentParser()
okt = Okt()

# general args
parser.add_argument("--mode", type=str,
                    help="pre or inc",
                    default='pre')

args = parser.parse_args()

def increase_entity_dict():
    ent2idx = pickle.load(open('../data/wiki/wiki_ent2idx_small.pickle', 'rb'))
    entity_dict = pickle.load(open('../data/wiki/wiki_entity_dict.pickle', 'rb'))
    redirects = pickle.load(open('../data/wiki/redirects.pickle', 'rb'))
    with open('../data/wiki/wiki_mention.tsv', 'r', encoding='utf-8') as f:
        l = 0
        m = 0
        for line in f:
            key = line.split('\t')[0]
            val = line.split('\t')[1][:-1]
            try:
                ent2idx[key]
            except KeyError:
                key = redirects[key]

            try:
                target = entity_dict[val]
                try:
                    target[key]
                except KeyError:
                    entity_dict[val][key] = 1
                    l += 1
            except KeyError:
                entity_dict[val] = {}
                entity_dict[val][key] = 1
                m += 1
            
        print(l)
        print(m)

    with open('../data/wiki/large_entity_dict.pickle', 'wb') as f:
        pickle.dump(entity_dict, f, pickle.HIGHEST_PROTOCOL)

    return entity_dict

def increase_calc_dict(entity_dict):
    calc_dict = {}
    idx = 1
    for m, e in entity_dict.items():
        print(idx)
        values = basic_prob(list(e.values()))
        calc_dict[m] = {}
        for i, (key, value) in enumerate(e.items()):
            calc_dict[m][key] = (values[i], idx)
            idx += 1

    with open('../data/wiki/large_entity_calc.pickle', 'wb') as f:
        pickle.dump(calc_dict, f, pickle.HIGHEST_PROTOCOL)

    return calc_dict

def add_unk(entity_dict):
    for key in entity_dict.keys():
        entity_dict[key]['#UNK#'] = 0

    calc_dict = {}
    idx = 1
    for m, e in entity_dict.items():
        print(idx)
        values = unk_prob(list(e.values()))
        calc_dict[m] = {}
        for i, (key, value) in enumerate(e.items()):
            calc_dict[m][key] = (values[i], idx) if key != '#UNK#' else (0.1, idx)
            idx += 1

    with open('../data/wiki/unk_entity_calc.pickle', 'wb') as f:
        pickle.dump(calc_dict, f, pickle.HIGHEST_PROTOCOL)

def unk_prob(x):
    x = np.array(x, dtype=np.float)
    return np.around(x * 0.9 / np.sum(x), 4)

def main():
    if args.mode == "inc":
        try:
            entity_dict = pickle.load(open('../data/wiki/large_entity_dict.pickle', 'rb'))
        except FileNotFoundError:
            entity_dict = increase_entity_dict()

        try:
	        pass
        except FileNotFoundError:
	        pass

        add_unk(entity_dict)
    elif args.mode == "pre":
        rmpunc = re.compile('[^a-zA-Z0-9ㄱ-ㅣ가-힣]+')
        with open('../data/wiki/wiki_mention.tsv', 'r', encoding='utf-8') as f:
            with open('../data/wiki/target_mention.tsv', 'w', encoding='utf-8') as g:
                l = 0
                for line in f:
                    key, value = line[:-1].split('\t')
                    key = rmpunc.sub('', key)
                    value = rmpunc.sub('', value)
                    if key in value or value in key:
                        continue
                    key_list = [k for k in key]
                    val_list = [v for v in value]
                    co_list = list(filter(lambda x: x in val_list, key_list))
                    ratio = len(co_list) / len(key_list)
                    if ratio > 0.5:
                        continue
                    g.write(line)
                    l += 1

                print(l)

if __name__ == "__main__":
    main()
