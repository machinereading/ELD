import sys
import configparser
import argparse
import os

import tensorflow as tf

from src.ner.NERMain import NER
from src.el.ELMain import EL
from src import GlobalValues as gl
from src.utils.embedding import load_embedding as Embedding

class PropClass():
	pass

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["train", "eval", "predict"], required=True)
parser.add_argument("--target_dir", type=str, required=True)
args = parser.parse_args()

if not args.target_dir.endswith("/"):
	args.target_dir += "/"
gl.target_dir = args.target_dir
run_config = None
for item in os.listdir(args.target_dir):
	if item.endswith(".ini"):
		run_config = configparser.ConfigParser()
		run_config.read(args.target_dir+"/"+item)
		break

if run_config is None:
	print("Please make config file and make its ending with .ini")
	sys.exit(1)
setattr(gl, "run_mode", args.mode)
gpu_limit = None

if "gpu_setting" in run_config:
	os.environ["CUDA_VISIBLE_DEVICES"] = run_config["gpu_setting"]["device"]
	gpu_limit = tf.GPUOptions(per_process_gpu_memory_fraction=float(run_config["gpu_setting"]["limit"]))


# write config data into globalvalue object
for k in run_config:
	setattr(gl, k, PropClass())
	for kk, vv in run_config[k].items():
		try:
			vv = float(vv)
		except:
			try:
				vv = int(vv)
			except:
				pass
		
		setattr(getattr(gl, k), kk, vv)

# print(gl.data.data_drop_rate)

# set embedding
print("====Loading embedding====")

gl.embedding = Embedding(run_config["embedding"]["embedder"])(run_config["embedding"]["embedding_file"])

if "char_embedder" in run_config["embedding"]:
	gl.char_embedding = Embedding(run_config["embedding"]["char_embedder"])(run_config["embedding"]["char_embedding_file"])


# if "data" in run_config:
# 	setattr(gl, "data", PropClass())
# 	for k, v in run_config["data"]:
# 		setattr(gl.data, k, v)

# print("====Embedding loaded!====")

if "ner" in run_config:
	print("====Running NER====")
	if "run_etri" in run_config["ner"] and gl.boolmap[run_config["ner"]["run_etri"]]:
		pass
	else:
		setattr(gl, "ner", PropClass())
		for k, v in run_config["ner"].items():
			try:
				v = int(v)
			except Exception:
				pass
			setattr(gl.ner, k, v)
		# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_limit)) if gpu_limit is not None else tf.Session()

		with (tf.Session(config=tf.ConfigProto(gpu_options=gpu_limit)) if gpu_limit is not None else tf.Session()) as sess:
			ner_module = NER(sess)
	print("====NER complete====")

if "el" in run_config:
	print("====Running EL====")