from src.eld.ELDMain import ELDNoDiscovery, DictBasedELD
from src.eld.utils import ELDArgs, Evaluator
from src.ds import Corpus
# a = ELDNoDiscovery("pred", ELDArgs())
from src.utils import jsondump, writefile

b = DictBasedELD("pred", ELDArgs())

corpus = Corpus.load_corpus("result.json")
run_result = b(corpus)
evaluator = Evaluator(ELDArgs(), b.data)
print(evaluator.evaluate_by_form([x.eld_pred_entity for x in run_result.eld_items], [x.entity for x in run_result.eld_items]))
writefile(["\t".join([x.eld_pred_entity, x.entity]) for x in run_result.eld_items], "ambiguous_dictbased_result.tsv")
