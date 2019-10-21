from src.el import EL
from src.utils import jsonload, jsondump
import os
el = EL("test", "no_okt_ent_emb")
d = "corpus/crowdsourcing_1903_formatted/"
testset = [jsonload(d+x) for x in os.listdir(d) if "_dev" not in x]
for item in testset:
	item["entities"] = [x for x in item["entities"] if x["dataType"] not in ["DATE", "JOB", "TIME", "QUANTITY"]]
result = el(*testset)
for item in result:
	item["entities"] = [x for x in item["entities"] if "entity" not in x or x["entity"] == "NOT_IN_CANDIDATE"]
	jsondump(item, "data/iterative/%s.json" % item["fileName"])
