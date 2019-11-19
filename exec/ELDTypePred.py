from src.ds import Corpus
from src.eld.modules.TypePred import TypeGiver
from src.eld.utils import TypeEvaluator
from src.utils import jsondump

corpus = Corpus.load_corpus("corpus/namu_eld_inputs_handtag_only/", limit=5000)
kbt = "data/eld/typerefer/entity_types.json"
t = "data/eld/typerefer/dbpedia_types"
d = ["data/eld/typerefer/domain.tsv", "data/eld/typerefer/inst-d.tsv", "data/eld/typerefer/inst-d-ms.tsv"]
r = ["data/eld/typerefer/range.tsv", "data/eld/typerefer/inst-r.tsv", "data/eld/typerefer/inst-r-ms.tsv"]
dr = [None, "data/eld/typerefer/inst-dr.tsv", "data/eld/typerefer/inst-dr-ms.tsv"]
names = ["ontology", "inst", "inst-ms"]
dr = [True, False]
ne = [True, False]
hierarchy = [True, False]
x = [[0], [1], [2]]

labels = TypeGiver(kbt, t, [], []).get_gold(*corpus.eld_items)
evaluator = TypeEvaluator()
preds = []
scores = []
result = [{}]
n = []

for use_dr in dr:
	for use_ne in ne:
		for use_hierarchy in hierarchy:
			if use_dr:
				for item in x:
					for mode in ["union", "intersect"]:
						dd = [d[y] for y in item]
						rr = [r[y] for y in item]
						name = "+".join([str(x) for x in [use_dr, use_ne, use_hierarchy, *[names[y] for y in item], mode]])
						typegiver = TypeGiver(kbt, t, dd, rr, use_dr=use_dr, use_ne=use_ne, use_hierarchy=use_hierarchy, mode=mode)
						pred = typegiver(*corpus.eld_items)
						preds.append(pred)
						score = evaluator(corpus.eld_items, pred, labels)
						result[0][name] = list(score)
						n.append(name)
						print(score)
			else:
				name = "+".join([str(x) for x in [use_dr, use_ne, use_hierarchy]])
				typegiver = TypeGiver(kbt, t, use_dr=use_dr, use_ne=use_ne, use_hierarchy=use_hierarchy)
				pred = typegiver(*corpus.eld_items)
				preds.append(pred)
				score = evaluator(corpus.eld_items, pred, labels)
				result[0][name] = list(score)
				n.append(name)
				print(score)

# for item in x:
# 	for mode in ["union", "intersect"]:
# 		dd = [d[y] for y in item]
# 		rr = [r[y] for y in item]
# 		name = "+".join([names[y] for y in item] + [mode])
# 		typegiver = TypeGiver(kbt, t, dd, rr, use_dr=True, mode=mode)
# 		pred = typegiver(*corpus.eld_items)
# 		preds.append(pred)
# 		score = evaluator(corpus.eld_items, pred, labels)
# 		result[0][name] = list(score)
# 		n.append(name)
# 		print(score)

for x in zip(corpus.eld_items, labels, *preds):
	try:
		d = {
			"entity"  : x[0].entity,
			"sentence": x[0].parent_sentence.original_sentence,
			"relation": [[y.relation, y.outgoing, x[0].parent_sentence.entities[x[0].entity_idx + y.relative_index].entity, y.score] for y in x[0].relation],
			"answer"  : x[1]
		}
		for i, nn in enumerate(n):
			d[nn] = x[2+i]
		result.append(d)
	except:
		continue
jsondump(result, "type_analysis_with_dr.json")
