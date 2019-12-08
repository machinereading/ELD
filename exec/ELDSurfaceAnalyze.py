from src.utils import jsonload, jsondump

model_name = "pred_with_neg_softmax_loss"
target_dir = "eld_test/%s/" % model_name
analysis_target = "pred_test_cand_ambset"
# j = jsonload(target_dir + analysis_target + ".json")
j = jsonload("runs/eld/pred_with_neg_namu_word_emb/pred_with_neg_namu_word_emb_34.json")
mt = 0
ms = 0
for k, v in j["score"].items():
	s = v["Out-KB"][-1]
	if s > ms:
		mt = k[:4]
		ms = s

surface_pred_dict = {}
gold_pred_dict = {}
for item in j["data"]:
	gold = item["Entity"]
	if not gold.startswith("namu_"): continue
	pred = item["EntPred"][mt].split(":")[0]
	surface = item["Surface"]
	pg = "%s:%s" % (pred, gold)
	if surface not in surface_pred_dict:
		surface_pred_dict[surface] = {}
	if pg not in surface_pred_dict[surface]:
		surface_pred_dict[surface][pg] = 0
	surface_pred_dict[surface][pg] += 1
	if gold not in gold_pred_dict:
		gold_pred_dict[gold] = {}
	if pred not in gold_pred_dict[gold]:
		gold_pred_dict[gold][pred] = 0
	gold_pred_dict[gold][pred] += 1

jsondump(surface_pred_dict, "pred_with_neg_namu_word_emb_spd.json")
jsondump(gold_pred_dict, "pred_with_neg_namu_word_emb_gpd.json")

