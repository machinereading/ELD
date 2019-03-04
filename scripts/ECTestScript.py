from src.ec.ECMain import EC
from src.utils import readfile, writefile, jsondump, jsonload, TimeUtil


module = EC()

j = jsonload("corpus/dark_entity_candidates.json")
r = []
for i, item in enumerate(j):
	r.append("c%d {%s}" % (i, item["surface"]))


# x = readfile("corpus/de_set.set")
# r = []
# for line in x:
# 	r.append("c%s" % line)

writefile(r, "corpus/de_set.set")

jsondump(module.cluster("corpus/de_set.set"), "ec_results/ec_result.json")

TimeUtil.time_analysis()
