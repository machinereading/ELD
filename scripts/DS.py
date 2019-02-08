
from src.el.ELMain import EL
from src.ec.utils.train_set_generation import generate_trainset
from src.utils import jsonload, writefile, readfile, TimeUtil
import json
import os
# with TimeUtil.TimeChecker("EC_DataInit"):
# 	el_module = EL("test", "model")
# 	rep = lambda bold: bold.replace("\"BOLD\"", "").replace("\"/BOLD\"", "")
# 	corpus_dir = "corpus/kowiki_output_json_plain/"
# 	corpus = readfile("corpus/ec_corpus.txt")
# 	print(len(corpus))

with TimeUtil.TimeChecker("EC_DS"):
	generate_trainset("../ref/mulrel-nel/el_result")

TimeUtil.time_analysis()