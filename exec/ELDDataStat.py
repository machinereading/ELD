from src.ds import Corpus
from src.utils import readfile

corpus_dir = "corpus/namu_eld_inputs_handtag_only/"
train_filter = "corpus/namu_handtag_only_train"
dev_filter = "corpus/namu_handtag_only_dev"
test_filter = "corpus/namu_handtag_only_test"
corpus = Corpus.load_corpus(corpus_dir)
for f in [train_filter, dev_filter, test_filter]:
	eld_items = []
	filter_entities = [x for x in readfile(f)]
	for token in corpus.eld_items:
		if token.entity in filter_entities:
			eld_items.append(token)
	out_kb = list(filter(lambda x: x.entity.startswith("namu_"), eld_items))
	print(len(eld_items), len(eld_items) - len(out_kb), len(out_kb))
