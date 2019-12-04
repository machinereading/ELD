from src.ds import Corpus
from src.eld.DiscoveryOnly import DiscoveryModel
from src.utils import jsondump

corpus1 = Corpus.load_corpus("corpus/nokim.json")
corpus2 = Corpus.load_corpus("corpus/namu_eld_handtag_test2/")
filter_entities = ["namu_읍내", "namu_음악캠프"]
for item in corpus2.eld_items:
	if item.entity in filter_entities:
		item.target = False


model1 = DiscoveryModel("test", "discovery_degree_surface_4")

for model in [model1]:
	j1 = model.test(corpus1)
	j2 = model.test(corpus2)
	jsondump(j1, "discovery_amb_%s.json" % model.model_name)
	jsondump(j2, "discovery_test_%s.json" % model.model_name)
