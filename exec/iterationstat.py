from src.utils import jsonload, writefile

result = []
result_dark_only = []
for name in ["ev_none", "ev_all", "model_nn_explicit"]:
	j = jsonload("data/iterative_el_result_%s_with_fake.json" % name)
	sb = []
	nb = []
	b = []
	dsb = []
	dnb = []
	db = []
	for sent in j:
		for ent in sent["entities"]:
			if "answer" not in ent:
				continue
			if name == "ev_none":
				sb.append(ent["surface"])
				nb.append(ent["answer"])
			b.append(ent["predict"])
			if "manual_entity" in ent:
				if name == "ev_none":
					dnb.append(ent["surface"])
					dnb.append(ent["manual_entity"])
				db.append(ent["predict"])
	if name == "ev_none":
		result.append(sb)
		result.append(nb)
		result_dark_only.append(dsb)
		result_dark_only.append(dnb)
	result.append(b)
	result_dark_only.append(db)

print(len(result[0]), len(result_dark_only[0]))
rresult = []
better = lambda answer, baseline, prediction: "Better" if answer != baseline and answer == prediction else "Worse" if answer == baseline and answer != prediction else "Same"
for s, e, n, a, m in zip(*result):
	rresult.append([s, e, n, a, m, "O" if n == m else "X", "O" if e == m else "X", better(e, n, m)])
rresult_dark = []
for s, e, n, a, m in zip(*result_dark_only):
	rresult_dark.append([s, e, n, a, m, "O" if n == m else "X", "O" if e == m else "X", better(e, n, m)])


# result = [*zip(*result)]
# result_dark_only = [*zip(*result_dark_only)]
# print(result[0])
writefile(["\t".join(x) for x in rresult], "data/iterative_stat_all.tsv")
writefile(["\t".join(x) for x in rresult_dark], "data/iterative_stat_dark.tsv")

