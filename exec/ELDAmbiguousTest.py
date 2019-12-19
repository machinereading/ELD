from src.ds import Corpus
from src.eld.DiscoveryOnly import DiscoveryModel
from src.eld.PredOnly import DictBasedPred, MSEEntEmbedding, NoRegister
from src.eld.utils import DataModule, ELDArgs
from src.utils import jsondump, jsonload
import numpy as np
import traceback
import os
corpus1 = Corpus.load_corpus("corpus/nokim_fixed.json")
for item in corpus1.eld_items:
	if not item.entity.startswith("namu_"):
		item.entity = "namu_" + item.entity
corpus2 = Corpus.load_corpus("corpus/namu_eld_handtag_test2/")
filter_entities = ["namu_읍내", "namu_음악캠프"]
for item in corpus2.eld_items:
	if item.entity in filter_entities:
		item.target = False
args = ELDArgs()
args.in_kb_linker = "mulrel"
# pred0 = NoRegister("train", "noreg", args=args)
# pred1 = DictBasedPred("test", "pred_with_neg_namu_word_emb")

# pred3 = MSEEntEmbedding("test", "pred_with_neg_softmax_no_char")
# pred4 = MSEEntEmbedding("test", "pred_with_neg_softmax_no_word_context")
# pred5 = MSEEntEmbedding("test", "pred_with_neg_softmax_no_entity_context")
# pred6 = MSEEntEmbedding("test", "pred_with_neg_softmax_no_type")
# pred2 = MSEEntEmbedding("test", "pred_mse_4")
# # pred3 = MSEEntEmbedding("test", "pred_mse_wo_modify_0")
# # discovery = DiscoveryModel("test", "discovery_degree_surface_4")
# pred2_1 = MSEEntEmbedding("test", "pred_mse_4")
# pred3_2 = MSEEntEmbedding("test", "pred_with_neg_sampling_4")
# pred4_1 = MSEEntEmbedding("test", "mse_pred_cand_only_with_neg_softmax_without_jamo_4")
#
# pred4_2 = MSEEntEmbedding("test", "mse_pred_cand_only_with_neg_softmax_without_jamo_4")
# pred4_2.data.cand_only = False
# pred4_3 = MSEEntEmbedding("test", "mse_pred_cand_only_with_neg_softmax_without_jamo_4")
# pred0 = MSEEntEmbedding("test", "pred_with_neg_namu_word_emb")
#
pred0 = MSEEntEmbedding("test", "pred_full_ffnn_transformer")
# pred1 = MSEEntEmbedding("test", "pred_with_neg_softmax_no_word_ffnn")
# pred2 = MSEEntEmbedding("test", "pred_with_neg_softmax_no_word_context_ffnn")
# pred3 = MSEEntEmbedding("test", "pred_with_neg_softmax_no_entity_context_ffnn")
# pred4 = MSEEntEmbedding("test", "pred_with_neg_softmax_no_type_ffnn")
# pred5 = MSEEntEmbedding("test", "pred_with_neg_softmax_no_char_ffnn")
# pred6 = MSEEntEmbedding("test", "pred_with_neg_softmax_no_rel_ffnn")
# pred4 = MSEEntEmbedding("test", " ")

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
test_nocand_nomod = True
test_cand_nomod = False
test_nocand_mod = False
test_cand_mod = False
# for model in [pred0, pred1, pred2, pred3, pred4, pred5, pred6]:
for model in [pred0]:
	target_dir = "eld_test/%s/" % model.model_name
	if not os.path.isdir(target_dir):
		os.mkdir(target_dir)
	if test_nocand_nomod:
		print("NOCAND, NOMOD")
		j1 = model.test(corpus1)
		ms = 0
		mv = 0
		jsondump(j1, target_dir + "pred_test_nocand_ambset.json")
		try:
			for i, (s, v) in enumerate(j1["score"].items()):
				score = v["Total"][-1]
				if score > ms:
					ms = score
					mv = i + 1
			np.save(target_dir + "cache_emb_nocand_ambset.npy", model.data.cache_entity_embedding[mv])
			jsondump([list(x) for x in model.data.cache_entity_surface_dict[mv]], target_dir + "cache_dict_nocand_ambset.json")
		except:
			traceback.print_exc()
		j2 = model.test(corpus2)
		jsondump(j2, target_dir + "pred_test_nocand_testset.json" )
		# score1 = j1["score"] if type(j1) is dict else j1
		ms = 0
		mv = 0
		try:
			for i, (s, v) in enumerate(j1["score"].items()):
				score = v["Total"][-1]
				if score > ms:
					ms = score
					mv = i + 1
			np.save(target_dir + "cache_emb_nocand_testset.npy", model.data.cache_entity_embedding[mv])
			jsondump([list(x) for x in model.data.cache_entity_surface_dict[mv]], target_dir + "cache_dict_nocand_testset.json" )
		except:
			traceback.print_exc()

	if test_cand_nomod:
		print("CAND, NOMOD")
		model.data.cand_only = True

		j1 = model.test(corpus1)
		ms = 0
		mv = 0
		jsondump(j1, target_dir + "pred_test_cand_ambset.json")
		try:
			for i, (s, v) in enumerate(j1["score"].items()):
				score = v["Total"][-1]
				if score > ms:
					ms = score
					mv = i + 1
			np.save(target_dir + "cache_emb_cand_ambset.npy", model.data.cache_entity_embedding[mv])
			jsondump([list(x) for x in model.data.cache_entity_surface_dict[mv]], target_dir + "cache_dict_cand_ambset.json")
		except:
			traceback.print_exc()
		j2 = model.test(corpus2)
		jsondump(j2, target_dir + "pred_test_cand_testset.json")
		# score1 = j1["score"] if type(j1) is dict else j1
		ms = 0
		mv = 0
		try:
			for i, (s, v) in enumerate(j1["score"].items()):
				score = v["Total"][-1]
				if score > ms:
					ms = score
					mv = i + 1
			np.save(target_dir + "cache_emb_cand_testset.npy", model.data.cache_entity_embedding[mv])
			jsondump([list(x) for x in model.data.cache_entity_surface_dict[mv]], target_dir + "cache_dict_cand_testset.json")
		except:
			traceback.print_exc()

	if test_nocand_mod:
		print("NOCAND, MOD")
		model.data.cand_only = False
		model.data.modify_entity_embedding = True
		j1 = model.test(corpus1)
		ms = 0
		mv = 0
		jsondump(j1, target_dir + "pred_test_nocand_mod_ambset.json")
		try:
			for i, (s, v) in enumerate(j1["score"].items()):
				score = v["Total"][-1]
				if score > ms:
					ms = score
					mv = i + 1
			np.save(target_dir + "cache_emb_nocand_mod_ambset.npy", model.data.cache_entity_embedding[mv])
			jsondump([list(x) for x in model.data.cache_entity_surface_dict[mv]], target_dir + "cache_dict_nocand_mod_ambset.json")
		except:
			traceback.print_exc()
		j2 = model.test(corpus2)
		jsondump(j2, target_dir + "pred_test_nocand_mod_testset.json")
		# score1 = j1["score"] if type(j1) is dict else j1
		ms = 0
		mv = 0
		try:
			for i, (s, v) in enumerate(j1["score"].items()):
				score = v["Total"][-1]
				if score > ms:
					ms = score
					mv = i + 1
			np.save(target_dir + "cache_emb_nocand_mod_testset.npy", model.data.cache_entity_embedding[mv])
			jsondump([list(x) for x in model.data.cache_entity_surface_dict[mv]], target_dir + "cache_dict_nocand_mod_testset.json")
		except:
			traceback.print_exc()

	if test_cand_mod:
		print("CAND, MOD")
		model.data.cand_only = True
		j1 = model.test(corpus1)
		ms = 0
		mv = 0
		jsondump(j1, target_dir + "pred_test_cand_mod_ambset.json")
		try:
			for i, (s, v) in enumerate(j1["score"].items()):
				score = v["Total"][-1]
				if score > ms:
					ms = score
					mv = i + 1
			np.save(target_dir + "cache_emb_cand_mod_ambset.npy", model.data.cache_entity_embedding[mv])
			jsondump([list(x) for x in model.data.cache_entity_surface_dict[mv]], target_dir + "cache_dict_cand_mod_ambset.json")
		except:
			traceback.print_exc()
		j2 = model.test(corpus2)
		jsondump(j2, target_dir + "pred_test_cand_mod_testset.json")
		# score1 = j1["score"] if type(j1) is dict else j1
		ms = 0
		mv = 0
		try:
			for i, (s, v) in enumerate(j1["score"].items()):
				score = v["Total"][-1]
				if score > ms:
					ms = score
					mv = i + 1
			np.save(target_dir + "cache_emb_cand_mod_testset.npy", model.data.cache_entity_embedding[mv])
			jsondump([list(x) for x in model.data.cache_entity_surface_dict[mv]], target_dir + "cache_dict_cand_mod_testset.json")
		except:
			traceback.print_exc()
	# result1[model.model_name] = score1
	# result2[model.model_name] = score2
#
# jsondump(result1, "pred_test_ambset.json")
# jsondump(result2, "pred_test_testset.json")


