from src.eld.ELDMain import ELDNoDiscovery, DictBasedELD, VectorBasedELD
from src.eld.utils import ELDArgs, Evaluator, DataModule
from src.ds import Corpus
# a = ELDNoDiscovery("pred", ELDArgs())
from src.utils import jsondump, writefile
a = ELDNoDiscovery("pred", ELDArgs())
dictargs = ELDArgs()
dictargs.in_kb_linker = "pem"
dictargs.use_cache_kb = False
# dictargs.test_mode = True
b = DictBasedELD("pred", dictargs)
# c = VectorBasedELD("pred", "noattn_full_fixed")
# d = VectorBasedELD("pred", "full_with_surface")
# e = VectorBasedELD("pred", "full_with_degree")
data = DataModule("test", dictargs)
test_data = Corpus.load_corpus("corpus/namu_eld_handtag_test2/")
# for mod in [a,b,c,d,e]:
for mod in [a, b]:
	mod.data = data
	mod.evaluator = Evaluator(mod.args, data)
	print(mod.evaluator is not None)
	mod.test(test_data)
# corpus = Corpus.load_corpus("result_mapping.json")
# for i, mod in enumerate([a, b, c, d, e]):
# 	run_result = mod(corpus)
# 	evaluator = Evaluator(ELDArgs(), mod.data)
# 	print(evaluator.evaluate_by_form([x.eld_pred_entity for x in run_result.eld_items], [x.entity for x in run_result.eld_items]))
# 	writefile(["\t".join([x.eld_pred_entity, x.entity]) for x in run_result.eld_items], "ambiguous_dictbased_result_%s.tsv" % mod.model_name)
