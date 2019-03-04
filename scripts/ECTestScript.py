from src.ec.ECMain import EC
from src.utils import jsondump, jsonload, writefile
module = EC()

# set generation
de = jsonload("corpus/dark_entity_candidates.json")
r = []
for i, e in enumerate(de):
	r.append("%d {%s}" % (i, e["surface"]))
writefile(r, "corpus/de_set.set")

jsondump(module.cluster("corpus/de_set.set"), "ec_result.json")