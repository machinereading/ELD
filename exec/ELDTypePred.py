from src.ds import Corpus
from src.eld.modules.TypePred import TypeGiver
from src.eld.utils import TypeEvaluator
from src.utils import jsondump

corpus = Corpus.load_corpus("corpus/namu_eld_inputs_handtag_only/", limit=5000)
kbt = "data/eld/typerefer/entity_types.json"
t = "data/eld/typerefer/dbpedia_types"
d = ["data/eld/typerefer/domain.tsv", "data/eld/typerefer/inst-d.tsv", "data/eld/typerefer/inst-d-ms.tsv"]
r = ["data/eld/typerefer/range.tsv", "data/eld/typerefer/inst-r.tsv", "data/eld/typerefer/inst-r-ms.tsv"]
names = ["ontology", "inst", "inst-ms"]
x = [[0], [1],[2],[0,1],[0,2]]

labels = TypeGiver(kbt, t, [], []).get_gold(*corpus.eld_items)
evaluator = TypeEvaluator()
preds = []
scores = []
result = [{}]

for item in x:
	dd = [d[y] for y in item]
	rr = [r[y] for y in item]
	name = "+".join([names[y] for y in item])
	typegiver = TypeGiver(kbt, t, dd, rr)
	pred = typegiver(*corpus.eld_items)
	preds.append(pred)
	score = evaluator(corpus.eld_items, pred, labels)
	result[0][name] = list(score)
	print(score)

for x in zip(corpus.eld_items, labels, *preds):
	try:
		result.append({
			"entity": x[0].entity,
			"sentence": x[0].parent_sentence.original_sentence,
			"relation": [[y.relation, y.outgoing, x[0].parent_sentence.entities[x[0].entity_idx + y.relative_index].entity] for y in x[0].relation],
			"answer": x[1],
			"ontology": x[2],
			"inst": x[3],
			"inst-ms": x[4]
		})
	except: continue
jsondump(result, "type_analysis.json")
