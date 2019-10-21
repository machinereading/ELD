import argparse
import os

from src.eld import ELD
from src.eld.utils import ELDArgs
from src.utils.TimeUtil import time_analysis

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["train", "pred", "demo"], required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--transformer", type=str, default="separate", choices=["separate", "joint"])
parser.add_argument("--char_encoder", type=str, default="cnn", choices=["cnn", "selfattn"])
parser.add_argument("--word_encoder", type=str, default="cnn", choices=["cnn", "selfattn"])
# parser.add_argument("word_context_encoder")
parser.add_argument("--relation_encoder", type=str, default="cnn", choices=["cnn", "selfattn"])
parser.add_argument("--type_encoder", type=str, default="cnn", choices=["cnn", "selfattn"])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
args = parser.parse_args()
mode = args.mode
model_name = args.model_name

eld_args = ELDArgs(model_name)
eld_args.transformer_mode = args.transformer
eld_args.character_encoder = args.char_encoder
eld_args.word_encoder = args.word_encoder
eld_args.relation_encoder = args.relation_encoder
eld_args.type_encoder = args.type_encoder

module = ELD(mode, model_name, eld_args)

if mode == "train":
	module.train()
	time_analysis()
if mode == "pred":
	pass

if mode == "demo":
	pass


