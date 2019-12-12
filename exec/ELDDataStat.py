from src.ds import Corpus
from src.utils import readfile

corpus_dir = "corpus/namu_eld_inputs_handtag_only/"
train_filter = "corpus/namu_handtag_only_train"
dev_filter = "corpus/namu_handtag_only_dev"
test_filter = "corpus/namu_handtag_only_test"
train = Corpus.load_corpus("corpus/namu_eld_handtag_train2/")
dev = Corpus.load_corpus("corpus/namu_eld_handtag_dev2/")
test = Corpus.load_corpus("corpus/namu_eld_handtag_test2/")
amb = Corpus.load_corpus("corpus/nokim_fixed.json")
for name, corpus in zip(["train", "dev", "test", "amb"], [train, dev, test, amb]):
	se_dict = {}
	ec = {}
	ic = {}
	nic = {}
	for token in corpus.eld_items:
		target = nic if token.is_new_entity else ic
		if token.entity not in target:
			target[token.entity] = 0
		target[token.entity] += 1
	print(len(ic), len(nic), sum(ic.values()), sum(nic.values()), sum(ic.values()) / len(ic), sum(nic.values()) / len(nic))

	# out_kb = list(filter(lambda x: x.entity.startswith("namu_"), eld_items))
	# print(len(eld_items), len(eld_items) - len(out_kb), len(out_kb))
