import argparse

from src.eld import VectorBasedELD, BertBasedELD
from src.eld.utils import ELDArgs
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

parser.add_argument("--train_limit", type=int, default=-1)
parser.add_argument("--dev_limit", type=int, default=-1)
parser.add_argument("--modify_entity_embedding", action="store_true")
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
if model_name.startswith("bert"):
	module = BertBasedELD(mode, model_name, train_args=eld_args)
else:
	module = VectorBasedELD(mode, model_name, train_args=eld_args)

if mode == "train":
	module.train()
	time_analysis()
if mode == "pred":
	pass

if mode == "demo":
	pass

if mode == "typeeval":
	module.evaluate_type()

if mode == "test":
	module.test()
