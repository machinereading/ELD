from src.utils import jsonload, jsondump

# j1 = jsonload("runs/eld/nodiscovery_test.json")
j2 = jsonload("runs/eld/noattn_full/noattn_full_test.json")
result = {}
for item in j2["result"]:
	entity = item["Entity"]
	pred = item["EntPredClustered"]
	surface = item["Surface"]
	if entity not in result:
		result[entity] = {}
	if surface not in result[entity]:
		result[entity][surface] = {}
	if pred not in result[entity][surface]:

		result[entity][surface][pred] = 0
	result[entity][surface][pred] += 1

jsondump(result, "analysis_old.json")
# for ent, v in result.items():
# 	for surface, vv in v.items():
# 		for pred, vvv in vv.items():
#
