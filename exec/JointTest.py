from src.ds import Corpus
from src.eld.DiscoveryOnly import DiscoveryModel
from src.eld.PredOnly import DictBasedPred, MSEEntEmbedding, NoRegister
from src.eld.utils import DataModule, ELDArgs
from src.utils import jsondump, jsonload

def get_max_threshold(j):
	max_score = 0
	max_threshold = 0
	for k, v in j["score"].items():
		if v[2] > max_score:
			max_score = v[2]
			max_threshold = k
	return max_threshold

corpus1 = Corpus.load_corpus("corpus/nokim_fixed.json")
corpus2 = Corpus.load_corpus("corpus/namu_eld_handtag_test2/")
args = ELDArgs()
filter_entities = ["namu_읍내", "namu_음악캠프"]
for item in corpus2.eld_items:
	if item.entity in filter_entities:
		item.target = False
discovery_model = DiscoveryModel("test", "discovery_degree_surface_5")


print(len(corpus1.eld_items), len(corpus2.eld_items))

d1 = discovery_model.test(corpus1)
d2 = discovery_model.test(corpus2)

print(len(d1["data"]), len(d2["data"]))
t1 = get_max_threshold(d1)
t2 = get_max_threshold(d2)
d1 = [x["NewEntPred"] > t1 for x in d1["data"]]
d2 = [x["NewEntPred"] > t2 for x in d2["data"]]

pred_model = MSEEntEmbedding("test", "pred_with_neg_namu_word_emb")
p1 = pred_model.test(corpus1, d1)
p2 = pred_model.test(corpus2, d2)
