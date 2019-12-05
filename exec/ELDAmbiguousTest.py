from src.ds import Corpus
from src.eld.DiscoveryOnly import DiscoveryModel
from src.eld.PredOnly import DictBasedPred, MSEEntEmbedding, NoRegister
from src.eld.utils import DataModule, ELDArgs
from src.utils import jsondump, jsonload
import numpy as np
import traceback
corpus1 = Corpus.load_corpus("corpus/nokim_fixed.json")
corpus2 = Corpus.load_corpus("corpus/namu_eld_handtag_test2/")
filter_entities = ["namu_읍내", "namu_음악캠프"]
for item in corpus2.eld_items:
	if item.entity in filter_entities:
		item.target = False
args = ELDArgs()
args.in_kb_linker = "mulrel"
pred0 = NoRegister("train", "noreg", args=args)

# pred1 = DictBasedPred(DataModule("test", ELDArgs()))
# pred2 = MSEEntEmbedding("test", "pred_mse_4")
# pred3 = MSEEntEmbedding("test", "pred_mse_wo_modify_0")
# discovery = DiscoveryModel("test", "discovery_degree_surface_4")
pred2_1 = MSEEntEmbedding("test", "pred_mse_4")
pred3_2 = MSEEntEmbedding("test", "pred_with_neg_sampling_4")
pred4_1 = MSEEntEmbedding("test", "mse_pred_cand_only_with_neg_softmax_without_jamo_4")

pred4_2 = MSEEntEmbedding("test", "mse_pred_cand_only_with_neg_softmax_without_jamo_4")
pred4_2.data.cand_only = False
pred4_3 = MSEEntEmbedding("test", "mse_pred_cand_only_with_neg_softmax_without_jamo_4")
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
for model in [pred0, pred3, pred4, pred5]:
# for model in [pred1, pred2, pred3, pred4, pred5]:
	j1 = model.test(corpus1)
	ms = 0
	mv = 0
	jsondump(j1, "eld_test/pred_test_ambset_%s.json" % model.model_name)
	try:
		for i, (s, v) in enumerate(j1["score"].items()):
			score = v["Total"][-1]
			if score > ms:
				ms = score
				mv = i + 1
		np.save("eld_test/%s_cache_emb_ambset.npy" % model.model_name, model.data.cache_entity_embedding[mv])
		jsondump([list(x) for x in model.data.cache_entity_surface_dict[mv]], "eld_test/%s_cache_dict_ambset.json" % model.model_name)
	except:
		traceback.print_exc()
	j2 = model.test(corpus2)
	jsondump(j2, "eld_test/pred_test_testset_%s.json" % model.model_name)
	# score1 = j1["score"] if type(j1) is dict else j1
	ms = 0
	mv = 0
	try:
		for i, (s, v) in enumerate(j1["score"].items()):
			score = v["Total"][-1]
			if score > ms:
				ms = score
				mv = i + 1
		np.save("eld_test/%s_cache_emb_ambset.npy" % model.model_name, model.data.cache_entity_embedding[mv])
		jsondump([list(x) for x in model.data.cache_entity_surface_dict[mv]], "eld_test/%s_cache_dict_ambset.json" % model.model_name)
	except:
		traceback.print_exc()
	del model
	# result1[model.model_name] = score1
	# result2[model.model_name] = score2
#
# jsondump(result1, "pred_test_ambset.json")
# jsondump(result2, "pred_test_testset.json")


