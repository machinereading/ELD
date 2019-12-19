from src.utils import jsonload, jsondump

for item in ["testset", "ambset"]:
	j = jsonload("eld_test/pred_with_neg_softmax_no_rel_ffnn/pred_test_nocand_%s.json" % item)
	se = {}
	es = {}
	for ent in j["data"]:
		surface = ent["Surface"]
		entity = ent["Entity"]
		if surface not in se:
			se[surface] = {}
		if entity not in se[surface]:
			se[surface][entity] = 0
		se[surface][entity] += 1
		if entity not in es:
			es[entity] = {}
		if surface not in es[entity]:
			es[entity][surface] = 0
		es[entity][surface] += 1
	jsondump(se, "se_%s.json" % item)
	jsondump(es, "es_%s.json" % item)

