from src.utils import jsonload

filters = ["char", "word", "word_context", "entity_context", "type", "rel"]
models = ["pred_full_ffnn_transformer", *["pred_with_neg_softmax_no_%s_ffnn" % x for x in filters]]
# targets = ["pred_test_nocand_ambset", "pred_test_cand_ambset", "pred_test_nocand_mod_ambset", "pred_test_cand_mod_ambset"]
# targets = ["pred_test_nocand_%s" % x for x in ["testset", "ambset"]]
target_dir = "eld_test2"
testset = [jsonload("%s/noemb/pred_test_cand_testset.json" % target_dir)] + list(map(lambda x: jsonload("%s/%s/%s.json" % (target_dir, x, "pred_test_nocand_testset")), models))
ambset = [jsonload("%s/noemb/pred_test_cand_ambset.json" % target_dir)] + list(map(lambda x: jsonload("%s/%s/%s.json" % (target_dir, x, "pred_test_nocand_ambset")), models))
# js = list(map(lambda x: jsonload(target_dir + x + ".json"), targets))

def get_maximum_threshold(j):
	ms = 0
	mt = 0
	for k, score in j["score"].items():
		s = score["Out-KB"][-1]
		if s > ms:
			ms = s
			mt = "%.2f" % float(k)
	return mt
for js, result_file_name in [(testset, "testset"), (ambset, "ambset")]:
	max_scores = list(map(get_maximum_threshold, js))
	with open("%s/%s.tsv" % (target_dir, result_file_name), "w", encoding="UTF8") as f:
		f.write("\t".join(["Surface", "Entity", "Context", "noemb", "full"] + ["-"+x for x in filters]) + "\n")
		for item in zip(*[x["data"] for x in js]):

			pred = [x["EntPred"][k].split(":")[0] for k, x in zip(max_scores, item)]
			entity = item[0]["Entity"]
			surface = item[0]["Surface"]
			context = item[0]["Context"]
			f.write("\t".join([surface, entity, context] + pred) + "\n")
	with open("%s/%s_score.tsv" % (target_dir, result_file_name), "w", encoding="UTF8") as f:
		f.write("\t".join(["", "In", "Out", "Total", "ARI"])+"\n")
		for item, name, ms in zip(js, ["NoEmb", "All", *["-"+x for x in filters]], max_scores):
			score = [*[item["score"][ms][x][-1] for x in ["In-KB", "Out-KB", "Total"]], item["score"][ms]["ARI"]]
			f.write("\t".join([name, str(score[0]), str(score[1]), str(score[2]), str(score[3])])+"\n")

