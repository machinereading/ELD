# import endpoint.
# stores only global values
# set attributes by calling setattr in main module
import logging
import re
import os
from datetime import datetime

from .utils import readfile

# logging config
formatter = logging.Formatter('%(levelname)s: %(asctime)s %(message)s', datefmt='%I:%M:%S')
logging.basicConfig(format='%(levelname)s: %(asctime)s %(message)s', datefmt='%I:%M:%S')
logger = logging.getLogger("DefaultLogger")
logger.setLevel(logging.DEBUG)
if not os.path.isdir("log"): os.mkdir("log")
f = logging.FileHandler("log/run_%s.log" % re.sub(r"[ :/]", "_", str(datetime.now())[:-7]))
# c = logging.StreamHandler()

f.setLevel(logging.DEBUG)
f.setFormatter(formatter)

# c.setLevel(logging.INFO)
# c.setFormatter(formatter)
# c.propagate = False
# logger.addHandler(c)
logger.addHandler(f)
logger.info("[START]")

boolmap = {"True": True, "False": False}
corpus_home = "corpus/"
entity_id_map = {}
for i, k in enumerate(readfile("data/el/embeddings/dict.entity")):
	entity_id_map[k.split("\t")[0].replace("ko.dbpedia.org/resource/", "")] = i
# entity_id_map = {k.split("\t")[0].replace("ko.dbpedia.org/resource/", ""): i for i, k in enumerate([x for x in readfile("data/el/embeddings/dict.entity")])}
id_entity_map = {v: k for k, v in entity_id_map.items()}
