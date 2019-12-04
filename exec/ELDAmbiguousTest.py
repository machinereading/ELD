from src.ds import Corpus
from src.eld.DiscoveryOnly import DiscoveryModel
from src.eld.PredOnly import DictBasedPred, MSEEntEmbedding, NoRegister
from src.eld.utils import DataModule, ELDArgs
from src.utils import jsondump, jsonload
import numpy as np

corpus1 = Corpus.load_corpus("corpus/nokim_fixed.json")
corpus2 = Corpus.load_corpus("corpus/namu_eld_handtag_test2/")
filter_entities = ["namu_읍내", "namu_음악캠프"]
for item in corpus2.eld_items:
	if item.entity in filter_entities:
		item.target = False
pred0 = NoRegister("test", "mse_pred_cand_only_with_neg_softmax_without_jamo_4")

# pred1 = DictBasedPred(DataModule("test", ELDArgs()))
# pred2 = MSEEntEmbedding("test", "pred_mse_4")
# pred3 = MSEEntEmbedding("test", "pred_mse_wo_modify_0")
# discovery = DiscoveryModel("test", "discovery_degree_surface_4")
pred3 = MSEEntEmbedding("test", "pred_mse_wo_modify_with_neg_4")
pred4 = MSEEntEmbedding("test", "mse_pred_cand_only_with_neg_softmax_without_jamo_4")
pred4.data.cand_only = False
# pred5 = MSEEntEmbedding("test", "pred_with_neg_sampling_4")
# pred6 = MSEEntEmbedding("test", "mse_pred_cand_only_with_neg_softmax_4")
# d = jsonload("runs/eld/discovery_degree_surface_4/discovery_degree_surface_4_test.json")
# is_dark_entity = [x["NewEntPred"] > 0.6 for x in d["data"]]

# dr = discovery(corpus2)
# is_dark_entity = [x.is_dark_entity for x in dr.eld_items]
# j = pred5.test(corpus2)
# jsondump(j, "jointtest.json")

# pred5 = NoRegister("test", "pred_mse_4")
result1 = {}
result2 = {}
for model in [pred0, pred3, pred4]:
# for model in [pred1, pred2, pred3, pred4, pred5]:
	j1 = model.test(corpus1)
	ms = 0
	mv = 0
	jsondump(j1, "eld_test/pred_test_ambset_%s.json" % model.model_name)
	try:
		for s, v in j1["score"].items():
			score = v["Total"][-1]
			if score > ms:
				ms = score
			mv = int(s)
		np.save("eld_test/%s_cache_emb_ambset.npy" % model.model_name, model.data.cache_entity_embedding[mv])
		jsondump([list(x) for x in model.data.cache_entity_surface_dict[mv]], "eld_test/%s_cache_dict_ambset.json" % model.model_name)
	except:
		pass
	j2 = model.test(corpus2)
	ms = 0
	mv = 0
	for s, v in j1["score"].items():
		score = v["Total"][-1]
		if score > ms:
			ms = score
		mv = int(s)
	jsondump(j2, "eld_test/pred_test_testset_%s.json" % model.model_name)
	# score1 = j1["score"] if type(j1) is dict else j1
	mt = 0
	mf = 0
	try:
		for k, v in j2["score"].items():
			if v["Total"][-1] > mf:
				mt = k
				mf = v["Total"][-1]
		np.save("eld_test/%s_cache_emb_testset.npy" % model.model_name, model.data.cache_entity_embedding[mv])
		jsondump([list(x) for x in model.data.cache_entity_surface_dict[mv]], "eld_test/%s_cache_dict_testset.json" % model.model_name)
	except: pass

	# result1[model.model_name] = score1
	# result2[model.model_name] = score2
#
# jsondump(result1, "pred_test_ambset.json")
# jsondump(result2, "pred_test_testset.json")


