from src.utils import jsonload, jsondump

ent_result = {}
suf_result = {}
pred_result = {}
j = jsonload("pred_train_mse_pred_cand_only_42.json")

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
	if pred not in pred_result:
		pred_result[pred] = {}
	if gold not in pred_result[pred]:
		pred_result[pred][gold] = 0
	pred_result[pred][gold] += 1
	ent_result[gold][surface][pred] += 1
	suf_result[surface][gold][pred] += 1

jsondump(pred_result, "pred_test_result2.json")
jsondump([ent_result, suf_result], "pred_test_analysis2.json")

same_micro = [0, 0]
d_micro = [0, 0]
same_macro = []
d_macro = []
smg = [0, 0]
dmg = [0, 0]
smag = []
dmag = []
for g, s in ent_result.items():
	ms = list(sorted(s.items(), key=lambda x: sum(x[1].values()), reverse=True))
	m = ms[0]
	c, t = 0, 0
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
	if len(ms) > 1:
		nm = ms[1:]
		for m in nm:
			c, t = 0, 0
			flag = False
			for k, v in m[1].items():
				if k == g:
					c += v
					flag = True
				t += v
			d_macro.append(c / t)
			d_micro[0] += c
			d_micro[1] += t
			if flag:
				dmg[0] += c
				dmg[1] += t
				dmag.append(c / t)


print(smg[0] / smg[1], same_micro[0] / same_micro[1], sum(smag) / len(smag), sum(same_macro) / len(same_macro))
print(dmg[0] / dmg[1], d_micro[0] / d_micro[1], sum(dmag) / len(dmag), sum(d_macro) / len(d_macro))

with open("clusters2.tsv", "w", encoding="UTF8") as f:
	for k, v in pred_result.items():
		f.write("\t".join([k, str(sum(v.values()))])+"\n")
with open("gold2.tsv", "w", encoding="UTF8") as f:
	for k, v in ent_result.items():
		s = 0
		for vv in v.values():
			s += sum(vv.values())
		f.write("\t".join([k, str(s)])+"\n")
