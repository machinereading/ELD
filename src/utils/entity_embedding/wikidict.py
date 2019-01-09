import re
import pickle
import numpy as np

# 데이터로부터 (mention, entity(link), type)의 리스트 생성
def make_entity_list():
    entity_list = []
    with open('../data/entityTypeTaggedText_fixed.txt', 'rb') as f:
        i = 1
        for d in f:
            print(i)
            text = d.decode('utf-8')
            r1 = re.compile("\[{3}[  ㄱ-ㅣ가-힣a-zA-Z0-9-_()\"\']+\|{1}[  ㄱ-ㅣ가-힣a-zA-Z0-9-_()\"\']+\]{3}<{2}[a-zA-Z]+\|{1}[A-Z]+>{2}")
            m1 = r1.finditer(text)
            for m in m1:
                parsed = text[m.start():m.end()]
                r2 = re.compile("\[{3}[  ㄱ-ㅣ가-힣a-zA-Z0-9-_()\"\']+\|{1}")
                m2 = r2.findall(parsed)
                mention = m2[0][3:-1]
                r3 = re.compile("\|{1}[  ㄱ-ㅣ가-힣a-zA-Z0-9-_()\"\']+\]{3}")
                m3 = r3.findall(parsed)
                entity = m3[0][1:-3]
                r4 = re.compile("<{2}[a-zA-Z]+\|{1}[A-Z]+>{2}")
                m4 = r4.findall(parsed)
                ent_type = m4[0][2:-2]
                entity_list.append((mention, entity, ent_type))
            i += 1

    with open('../data/wiki/wiki_entity.pickle', 'wb') as f:
        pickle.dump(entity_list, f, pickle.HIGHEST_PROTOCOL)

    return entity_list

# entity_list로부터 각 mention별 entity의 숫자를 기록한 dict로 변경
def make_entity_dict(entity_list):
    entity_dict = {}
    for m, e, _ in entity_list:
        try:
            target = entity_dict[m]
            try:
                _ = target[e]
                entity_dict[m][e] += 1
            except KeyError:
                entity_dict[m][e] = 1
        except KeyError:
            entity_dict[m] = {}
            entity_dict[m][e] = 1

    with open('../data/wiki/wiki_entity_dict.pickle', 'wb') as f:
        pickle.dump(entity_dict, f, pickle.HIGHEST_PROTOCOL)

    return entity_dict

# entity_dict로부터 확률값과 idx를 가진 dict로 변경
def make_calc_dict(entity_dict):
    calc_dict = {}
    idx = 1
    for m, e in entity_dict.items():
        print(idx)
        values = basic_prob(list(e.values()))
        calc_dict[m] = {}
        for i, (key, value) in enumerate(e.items()):
            calc_dict[m][key] = (values[i], idx)
            idx += 1

    with open('../data/wiki/wiki_entity_calc.pickle', 'wb') as f:
        pickle.dump(calc_dict, f, pickle.HIGHEST_PROTOCOL)

    return calc_dict        


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return np.around(e_x / e_x.sum(), 4)


def basic_prob(x):
    return np.around(x / np.sum(x), 4)


def main():
    try:
        entity_list = pickle.load(open('../data/wiki/wiki_entity.pickle', 'rb'))
    except FileNotFoundError:
        entity_list = make_entity_list()

    try:
        entity_dict = pickle.load(open('../data/wiki/wiki_entity_dict.pickle', 'rb'))
    except FileNotFoundError:
        entity_dict = make_entity_dict(entity_list)

    try:
        calc_dict = pickle.load(open('../data/wiki/wiki_entity_calc.pickle', 'rb'))
    except FileNotFoundError:
        calc_dict = make_calc_dict(entity_dict)

    keyword = 'E_(상수)'
    print(sorted(calc_dict[keyword].items(), key=lambda x: x[1][0], reverse=True))


if __name__ == "__main__":
    main()