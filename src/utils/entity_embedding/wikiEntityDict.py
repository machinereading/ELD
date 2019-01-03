import re
import pickle
import numpy as np
from konlpy.tag import Okt
from collections import Counter
from gensim.models import KeyedVectors
import argparse

parser = argparse.ArgumentParser()
okt = Okt()

# general args
parser.add_argument("--mode", type=str,
                    help="cooccur or unigram or all or small or dict or glove or w2v",
                    default='cooccur')

parser.add_argument("--step", type=int,
                    help="0 or 1",
                    default=0)

args = parser.parse_args()

def strip_e(st):
    RE_EMOJI = re.compile('[\U00010000-\U0010ffff]', flags=re.UNICODE)
    return RE_EMOJI.sub(r'', st)


def make_cooccurence():
    co_occur_dict = {}
    folder_list = ['AA', 'AB', 'AC', 'AD', 'AE', 'AF']
    for folder in folder_list:
        for i in range(100):
            print('{} of the docs in folder {}'.format(i, folder))
            page_str = str(i).zfill(2)
            dir = '../wikiextractor/text/{}/wiki_{}'.format(folder, page_str)
            print(dir)
            try:
                with open(dir, 'r', encoding='utf-8') as f:
                    text = ''.join([line.replace('\n', ' ') for line in f])
                    text_list = text.split('</doc>')
                    for text in text_list[:-1]:
                        parsed = text.split('">')
                        title = parsed[0].split('title="')[-1].replace(' ', '_')
                        desc = strip_e(''.join(parsed[1].split("  ")[1:-2]))
                        tags = list(filter(lambda x: x[1] not in ['Punctuation', 'Josa', 'Suffix'], okt.pos(desc)))
                        tags = ['/'.join(tag) for tag in tags]
                        if len(tags) > 0:
                            count = dict(Counter(tags))
                            if len(count) >= 20:
                                co_occur_dict[title] = count
            except FileNotFoundError:
                print('{} folder is not exist'.format(dir))
                break

    with open('../data/wiki/wiki_entity_cooccur.pickle', 'wb') as f:
        pickle.dump(co_occur_dict, f, pickle.HIGHEST_PROTOCOL)


def make_all():
    unigram_dict = {}
    co_occur_dict = {}
    folder_list = ['AA', 'AB', 'AC', 'AD', 'AE', 'AF']
    for folder in folder_list:
        for i in range(100):
            print('{} of the docs in folder {}'.format(i, folder))
            page_str = str(i).zfill(2)
            dir = '../wikiextractor/text/{}/wiki_{}'.format(folder, page_str)
            print(dir)
            try:
                with open(dir, 'r', encoding='utf-8') as f:
                    text = ''.join([line.replace('\n', ' ') for line in f])
                    text_list = text.split('</doc>')
                    for text in text_list[:-1]:
                        parsed = text.split('">')
                        title = parsed[0].split('title="')[-1].replace(' ', '_')
                        desc = strip_e(''.join(parsed[1].split("  ")[1:-2]))
                        tags = list(filter(lambda x: x[1] not in ['Punctuation', 'Josa', 'Suffix'], okt.pos(desc)))
                        tags = ['/'.join(tag) for tag in tags]
                        if len(tags) > 0:
                            count = dict(Counter(tags))
                            co_occur_dict[title] = count
                        for tag in tags:
                            try:
                                count = unigram_dict[tag]
                                unigram_dict[tag] += 1
                            except KeyError:
                                unigram_dict[tag] = 1     
            except FileNotFoundError:
                print('{} folder is not exist'.format(dir))
                break

    with open('../data/wiki/wiki_entity_cooccur.pickle', 'wb') as f:
        pickle.dump(co_occur_dict, f, pickle.HIGHEST_PROTOCOL)

    with open('../data/wiki/wiki_entity_unigram.pickle', 'wb') as f:
        pickle.dump(unigram_dict, f, pickle.HIGHEST_PROTOCOL)


def make_small():
    unigram_dict = {}
    co_occur_dict = {}
    folder_list = ['AA', 'AB', 'AC', 'AD', 'AE', 'AF']
    for folder in folder_list:
        for i in range(100):
            print('{} of the docs in folder {}'.format(i, folder))
            page_str = str(i).zfill(2)
            dir = '../wikiextractor/text/{}/wiki_{}'.format(folder, page_str)
            print(dir)
            try:
                with open(dir, 'r', encoding='utf-8') as f:
                    text = ''.join([line.replace('\n', ' ') for line in f])
                    text_list = text.split('</doc>')
                    for text in text_list[:-1]:
                        parsed = text.split('">')
                        title = parsed[0].split('title="')[-1].replace(' ', '_')
                        desc = strip_e(' '.join(parsed[1].split("  ")[1:-2]))
                        desc = ' '.join(desc.split(' ')[:200])
                        tags = list(filter(lambda x: x[1] not in ['Punctuation', 'Josa', 'Suffix'], okt.pos(desc)))
                        tags = ['/'.join(tag) for tag in tags]
                        if len(tags) > 0:
                            count = dict(Counter(tags))
                            co_occur_dict[title] = count
                        for tag in tags:
                            try:
                                count = unigram_dict[tag]
                                unigram_dict[tag] += 1
                            except KeyError:
                                unigram_dict[tag] = 1     
            except FileNotFoundError:
                print('{} folder is not exist'.format(dir))
                break
    
    ent2idx = {}
    for i, (key, value) in enumerate(co_occur_dict.items()):
        ent2idx[key] = i

    word2idx = {}
    idx2freq = {}
    for i, (key, value) in enumerate(unigram_dict.items()):
        word2idx[key] = i
        idx2freq[i] = value

    with open('../data/wiki/wiki_entity_cooccur_small.pickle', 'wb') as f:
        pickle.dump(co_occur_dict, f, pickle.HIGHEST_PROTOCOL)

    with open('../data/wiki/wiki_entity_unigram_small.pickle', 'wb') as f:
        pickle.dump(unigram_dict, f, pickle.HIGHEST_PROTOCOL)

    with open('../data/wiki/wiki_ent2idx_small.pickle', 'wb') as f:
        pickle.dump(ent2idx, f, pickle.HIGHEST_PROTOCOL)

    with open('../data/wiki/wiki_word2idx_small.pickle', 'wb') as f:
        pickle.dump(word2idx, f, pickle.HIGHEST_PROTOCOL)

    with open('../data/wiki/wiki_idx2freq_small.pickle', 'wb') as f:
        pickle.dump(idx2freq, f, pickle.HIGHEST_PROTOCOL)


def make_ent2idx():
    ent2idx = {}
    calc_dict = pickle.load(open('../data/wiki/wiki_entity_cooccur.pickle', 'rb'))
    for i, (key, value) in enumerate(calc_dict.items()):
        ent2idx[key] = i

    print(len(ent2idx.items()))
    with open('../data/wiki/wiki_ent2idx.pickle', 'wb') as f:
        pickle.dump(ent2idx, f, pickle.HIGHEST_PROTOCOL)


def make_unigram():
    unigram_dict = {}
    folder_list = ['AA', 'AB', 'AC', 'AD', 'AE', 'AF']
    for folder in folder_list:
        for i in range(100):
            print('{} of the docs in folder {}'.format(i, folder))
            page_str = str(i).zfill(2)
            dir = '../wikiextractor/text/{}/wiki_{}'.format(folder, page_str)
            print(dir)
            try:
                with open(dir, 'r', encoding='utf-8') as f:
                    text = ''.join([line.replace('\n', ' ') for line in f])
                    text_list = text.split('</doc>')
                    for text in text_list[:-1]:
                        parsed = text.split('">')
                        title = parsed[0].split('title="')[-1].replace(' ', '_')
                        desc = strip_e(''.join(parsed[1].split("  ")[1:-2]))
                        tags = list(filter(lambda x: x[1] not in ['Punctuation', 'Josa', 'Suffix'], okt.pos(desc)))
                        tags = ['/'.join(tag) for tag in tags]
                        for tag in tags:
                            try:
                                count = unigram_dict[tag]
                                unigram_dict[tag] += 1
                            except KeyError:
                                unigram_dict[tag] = 1
            except FileNotFoundError:
                print('{} folder is not exist'.format(dir))
                break

    with open('../data/wiki/wiki_entity_unigram.pickle', 'wb') as f:
        pickle.dump(unigram_dict, f, pickle.HIGHEST_PROTOCOL)


def make_entity_dict():
    calc_dict = pickle.load(open('../data/wiki/wiki_ent2idx_small.pickle', 'rb'))
    calc_dict = list(sorted(calc_dict.items(), key=lambda x: x[1], reverse=False))
    with open('../data/wiki/dict.entity', 'w', encoding='utf-8') as f:
        for key, value in calc_dict:
            url = 'ko.dbpedia.org/resource/'
            line = url + key + '\t' + '1000' + '\n'
            f.write(line)


def make_unigram_freq():
    word2idx = {}
    idx2freq = {}
    unigram_dict = pickle.load(open('../data/wiki/wiki_entity_unigram.pickle', 'rb'))
    for i, (key, value) in enumerate(unigram_dict.items()):
        word2idx[key] = i
        idx2freq[i] = value

    with open('../data/wiki/wiki_word2idx.pickle', 'wb') as f:
        pickle.dump(word2idx, f, pickle.HIGHEST_PROTOCOL)

    with open('../data/wiki/wiki_idx2freq.pickle', 'wb') as f:
        pickle.dump(idx2freq, f, pickle.HIGHEST_PROTOCOL)


def make_glove_embedding():

    with open('../data/glove/vectors.txt', 'r', encoding='utf-8') as f:
        embs = []
        keys = []
        for line in f:
            temp = line.split()
            key = temp.pop(0)
            if len(temp) != 300:
                continue
            keys.append(key)
            embs.append(temp)

    with open('../data/wiki/glove/dict.word', 'w', encoding='utf-8') as f:
        for key in keys:
            line = key + '\t' + '1000' + '\n'
            f.write(line)

    np.save('../data/wiki/glove/word_embeddings.npy', np.array(embs, dtype=np.float32))


def make_w2v_embedding():
    wb = KeyedVectors.load("../data/wiki/w2v/w2v_wiki_300.bin")
    idx2word = wb.index2word
    embs = []
    keys = []
    for word in idx2word:
        emb = wb[word]
        embs.append(emb)
        keys.append(word)

    with open('../data/wiki/dict.word', 'w', encoding='utf-8') as f:
        for key in keys:
            line = key + '\t' + '1000' + '\n'
            f.write(line)

    np.save('../data/wiki/word_embeddings.npy', np.array(embs))

        
def main():
    if args.mode == "cooccur":
        if args.step == 0:
            make_cooccurence()
        elif args.step == 1:
            make_ent2idx()
    elif args.mode == "unigram":
        if args.step == 0:
            make_unigram()
        elif args.step == 1:
            make_unigram_freq()
    elif args.mode == "all":
        print("start all of data processing")
        make_all()
        print("stage 1 finished")
        make_ent2idx()
        print("stage 2 finished")
        make_unigram_freq()
        print("stage 3 finished")
    elif args.mode == "small":
        make_small()
    elif args.mode == "dict":
        make_entity_dict()
    elif args.mode == "glove":
        make_glove_embedding()
    elif args.mode == "w2v":
        make_w2v_embedding()
    else:   
        print("That mode is not existed")
    

if __name__ == "__main__":
    main()