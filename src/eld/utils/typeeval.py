from . import ELDArgs
from ...utils import jsonload

class TypeEvaluator:
	def __call__(self, ent, type_pred, type_label):
		assert len(ent) == len(type_pred) == len(type_label)
		top_items = ["Person", "Place", "Organisation", "Device", "Event", "Food", "MeanOfTransportation"]
		top_items = ["http://dbpedia.org/ontology/" + x for x in top_items]
		pred = 0
		recall = 0
		correct = 0
		top_total = 0
		top_correct = 0
		no_type_label = set([])
		no_type_pred = set([])
		ltl = []
		ptl = []
		for e, p, l in zip(ent, type_pred, type_label):
			# if len(l) == 0:
			# 	no_type_label.add(e)
			# if len(p) == 0:
			# 	no_type_pred.add(e)
			if len(l) == 0: continue
			ltl.append(len(l))
			ptl.append((len(p)))
			# print(len(p), len(l))
			pred += len(p)
			recall += len(l)
			correct += len(set(p) & set(l))
			top_label = set(filter(lambda x: x in top_items, l))
			top_pred = set(filter(lambda x: x in top_items, p))
			top_total += len(top_label)
			top_correct += len(top_label & top_pred)
		print(pred, recall, correct, top_total, top_correct)
		p = correct / pred if pred > 0 else 0
		r = correct / recall if recall > 0 else 0
		f1 = 2 * p * r / (p + r) if p + r > 0 else 0
		acc = top_correct / top_total if top_total > 0 else 0
		return p, r, f1, acc, sum(ltl) / len(ltl), sum(ptl) / len(ptl)
