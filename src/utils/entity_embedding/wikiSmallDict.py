import re
import pickle
import numpy as np
from konlpy.tag import Okt
from collections import Counter
import argparse

# parser = argparse.ArgumentParser()
okt = Okt()

# general args
# parser.add_argument("--mode", type=str,
#                     help="cooccur or unigram or all",
#                     default='cooccur')

# parser.add_argument("--step", type=int,
#                     help="0 or 1",
#                     default=0)

# args = parser.parse_args()

def strip_e(st):
    RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    return RE_EMOJI.sub(r'', st)

def main():
    with open('../data/wiki/wiki.tsv', 'r', encoding='utf-8') as wiki:
        cand_list = []
        for w in wiki:
            parsed = w.split('\t')
            parsed = parsed[parsed.index('CANDIDATES') + 1:parsed.index('GE:')]
            for p in parsed:
                cand = p.split(',')[-1]
                cand_list.append(cand)
        print(len(list(set(cand_list))))


if __name__ == "__main__":
    main()