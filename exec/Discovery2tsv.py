from src.utils import jsonload

model_name = "discovery_degree_surface_4"
j = jsonload("runs/eld/%s/%s_test.json" % (model_name, model_name))
with open("discovery_case_study.tsv", "w", encoding="UTF8") as f:
	for item in j["data"]:
		f.write("\t".join([item["Context"], item["Surface"], str(item["NewEntPred"]), str(item["NewEntLabel"])]) + "\n")
