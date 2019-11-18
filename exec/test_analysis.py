import os
from src.utils import jsonload, jsondump

test_models = ["full", "full_degree", "full_surface", "full_no_cache_kb"]
HOME = "runs/eld/"
avg = lambda x: sum(x) / len(x) if len(x) > 0 else 0
for model in test_models:
	target_files = []
	i = 0
	kbs = []
	ts = []
	ins = []
	outs = []

	result = []
	while True:
		fname = HOME + "%s/%s_%i_test.json" % (model, model, i)
		if os.path.isfile(fname):
			target_files.append(fname)
		else:
			break

	for j in map(jsonload, target_files):
		kbs.append(j["scores"]["KB expectation score"][-1])
		ts.append(j["scores"]["Total score"][0][-1])
		ins.append(j["scores"]["In-KB score"][0][-1])
		outs.append(j["scores"]["Out-KB score"][0][-1])
		if len(result) == 0:
			for item in j["result"]:
				result.append({
					"Surface": item["Surface"],
					"Context": item["Context"],
					"EntPred": [],
					"NewEntPred": [],
					"Entity": item["Entity"],
					"NewEnt": item["NewEntLabel"],
					"CorrectCount": 0
				})
		for res, r in zip(result, j["result"]):
			res["EntPred"].append(r["EntPredClustered"])
			res["NewEntPred"].append(r["NewEntPred"])
			if r["EntPredClustered"] == r["Entity"]:
				res["CorrectCount"] += 1
	d = {

	}
	jsondump(d, model+"_result_")
