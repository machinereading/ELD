from src.eld.ELDMain import ELDNoDiscovery, DictBasedELD, VectorBasedELD
from src.eld.utils import ELDArgs, Evaluator
from src.ds import Corpus
# a = ELDNoDiscovery("pred", ELDArgs())
from src.utils import jsondump, writefile
a = ELDNoDiscovery("pred", ELDArgs())
b = DictBasedELD("pred", ELDArgs())
# c = VectorBasedELD("pred", "")
corpus = Corpus.load_corpus("result.json")
for i, mod in enumerate([a, b]):
	run_result = mod(corpus)
	evaluator = Evaluator(ELDArgs(), mod.data)
	print(evaluator.evaluate_by_form([x.eld_pred_entity for x in run_result.eld_items], [x.entity for x in run_result.eld_items]))
	writefile(["\t".join([x.eld_pred_entity, x.entity]) for x in run_result.eld_items], "ambiguous_dictbased_result_%d.tsv" % i)
