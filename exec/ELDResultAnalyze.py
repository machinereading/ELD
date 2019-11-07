
from src.utils import jsonload, jsondump

models = ["full", "nochar", "noword", "nowordctx", "noentctx", "norel", "notype"]
js = []
for model in models:
	js.append(jsonload("runs/eld/noattn_%s/noattn_%s_best_eval.json" % (model, model)))
result = [["Surface", "Context", "Entity", "NewEnt"]]
for r in js[0]["result"]:
	result.append([r["Surface"], r["Context"], r["Entity"], str(r["NewEntLabel"])])

for m, j in zip(models, js):
	result[0] += ["%s_Ent" % m, "%s_newent" % m]
	entset = {}
	for i, r in enumerate(j["result"]):
		ep = r["EntPredClustered"]
		result[i+1] += [ep, str(r["NewEntPred"])]
		if r["NewEntLabel"] == 1 and ep not in ["EXPECTED_IN_KB_AS_OUT_KB", "CLUSTER_PREASSIGNED"]:
			if ep not in entset:
				entset[ep] = set([])
			entset[ep].add(r["Surface"])
	entset = {k: list(v) for k, v in entset.items()}
	jsondump(entset, m+"surface_analysis_eval.json")


with open("analysis_eval.tsv", "w", encoding="UTF8") as f:
	for r in result:
		f.write("\t".join(r)+"\n")
