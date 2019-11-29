from src.utils import jsonload, jsondump

ent_result = {}
suf_result = {}
j = jsonload("pred_test_mse_pred_cand_only_0.json")

for ent in j[0]["data"]:
	surface = ent["Surface"]
	gold = ent["Entity"]
	pred = ent["EntPred"]["6"].split(":")[0]
	if gold[:4] != "namu": continue
	if surface not in suf_result:
		suf_result[surface] = {}
	if gold not in suf_result[surface]:
		suf_result[surface][gold] = {}
	if pred not in suf_result[surface][gold]:
		suf_result[surface][gold][pred] = 0
	if gold not in ent_result:
		ent_result[gold] = {}
	if surface not in ent_result[gold]:
		ent_result[gold][surface] = {}
	if pred not in ent_result[gold][surface]:
		ent_result[gold][surface][pred] = 0
	ent_result[gold][surface][pred] += 1
	suf_result[surface][gold][pred] += 1

jsondump([ent_result, suf_result], "pred_test_analysis.json")

same_micro = [0, 0]
same_macro = []
smg = [0, 0]
smag = []
for g, s in ent_result.items():
	m = list(sorted(s.items(), key=lambda x: sum(x[1].values()), reverse=True))[0]
	c, t = 0, 0
	print(m)
	flag = False
	for k, v in m[1].items():
		if k == g:
			c += v
			flag = True
		t += v
	same_macro.append(c / t)
	same_micro[0] += c
	same_micro[1] += t
	if flag:
		smg[0] += c
		smg[1] += t
		smag.append(c / t)
print(smg[0] / smg[1], same_micro[0] / same_micro[1], sum(smag) / len(smag), sum(same_macro) / len(same_macro))
