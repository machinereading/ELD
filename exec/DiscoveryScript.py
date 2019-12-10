import argparse

from src.ds import Corpus
from src.eld.DiscoveryOnly import DiscoveryModel
from src.eld.utils import ELDArgs, DataModule
from src.utils.TimeUtil import time_analysis

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["train", "pred", "typeeval", "demo", "test"], required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--char_encoder", type=str, default="cnn", choices=["none", "cnn", "selfattn"])
parser.add_argument("--word_encoder", type=str, default="cnn", choices=["none", "cnn", "selfattn"])
parser.add_argument("--word_context_encoder", type=str, default="bilstm", choices=["none", "bilstm"])
parser.add_argument("--entity_context_encoder", type=str, default="bilstm", choices=["none", "bilstm"])
parser.add_argument("--relation_encoder", type=str, default="cnn", choices=["none", "cnn", "selfattn"])
parser.add_argument("--type_encoder", type=str, default="ffnn", choices=["none", "ffnn", "selfattn"])
parser.add_argument("--vector_transformer", type=str, default="cnn", choices=["cnn", "attn", "ffnn"])
parser.add_argument("--register_threshold", type=float, default=0.3)
parser.add_argument("--use_separate_feature_encoder", action="store_true")
parser.add_argument("--use_surface_info", action="store_true")
parser.add_argument("--use_candidate_info", action="store_true")
parser.add_argument("--use_kb_relation_info", action="store_true")
parser.add_argument("--no_use_cache_kb", action="store_true")
parser.add_argument("--train_limit", type=int, default=-1)
parser.add_argument("--dev_limit", type=int, default=-1)
parser.add_argument("--modify_entity_embedding", action="store_true")
parser.add_argument("--input_file", type=str)
parser.add_argument("--output_file", type=str)
parser.add_argument("--train_iter", type=int, default=1)
parser.add_argument("--code_test", action="store_true")
args = parser.parse_args()
mode = args.mode
model_name = args.model_name
eld_args = ELDArgs(model_name)
if mode == "typeeval":
	eld_args.type_prediction = True
if args.char_encoder == "none":
	eld_args.use_character_embedding = False
else:
	eld_args.character_encoder = args.char_encoder
if args.word_encoder == "none":
	eld_args.use_word_embedding = False
else:
	eld_args.word_encoder = args.word_encoder
if args.word_context_encoder == "none":
	eld_args.use_word_context_embedding = False
else:
	eld_args.word_context_encoder = args.word_context_encoder
if args.entity_context_encoder == "none":
	eld_args.use_entity_context_embedding = False
else:
	eld_args.entity_context_encoder = args.entity_context_encoder
if args.relation_encoder == "none":
	eld_args.use_relation_embedding = False
else:
	eld_args.relation_encoder = args.relation_encoder
if args.type_encoder == "none":
	eld_args.use_type_embedding = False
else:
	eld_args.type_encoder = args.type_encoder
eld_args.vector_transformer = args.vector_transformer
eld_args.new_ent_threshold = args.register_threshold
eld_args.train_corpus_limit = args.train_limit
eld_args.dev_corpus_limit = args.dev_limit
eld_args.modify_entity_embedding = args.modify_entity_embedding
eld_args.use_separate_feature_encoder = args.use_separate_feature_encoder
eld_args.use_surface_info = args.use_surface_info
eld_args.use_candidate_info = args.use_candidate_info
eld_args.use_kb_relation_info = args.use_kb_relation_info
eld_args.use_cache_kb = not args.no_use_cache_kb
eld_args.test_mode = args.code_test

if mode == "train":
	train_data = Corpus.load_corpus("corpus/namu_eld_handtag_train2/", limit=1500 if args.code_test else 0)
	# dev_data = Corpus.load_corpus("corpus/ambiguous_input.json", limit=500)
	dev_data = Corpus.load_corpus("corpus/namu_eld_handtag_dev2/", limit=1500 if args.code_test else 0)
	test_data = Corpus.load_corpus("corpus/namu_eld_handtag_test2/", limit=1500 if args.code_test else 0)
	if args.train_iter > 1:
		for i in range(args.train_iter):
			eld_args.model_name = "%s_%d" % (model_name, i)
			module = DiscoveryModel(mode, "%s_%d" % (model_name, i), args=eld_args)
			module.train(train_data, dev_data, test_data)
		import sys
		sys.exit(0)
	module = DiscoveryModel(mode, model_name, args=eld_args)
	module.train(train_data, dev_data, test_data)
if mode == "pred":
	try:
		import json
		with open(args.input_file, encoding="UTF8") as rf, open(args.output_file, "w", encoding="UTF8") as wf:
			items = json.load(rf)
			json.dump(module(*rf), wf, ensure_ascii=False, indent="\t")
	except:
		import traceback
		traceback.print_exc()

if mode == "demo":
	pass

if mode == "typeeval":
	module.evaluate_type()

if mode == "test":
	module.test()
