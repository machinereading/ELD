# import endpoint.
# stores only global values
# set attributes by calling setattr in main module
from .utils import readfile
import logging
import re
from datetime import datetime

# logging config
logger = logging.getLogger("DefaultLogger")
f = logging.FileHandler("log/run_%s.log" % re.sub(r"[ :/]", "_", str(datetime.now())[:-7]))
c = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(message)s', datefmt='%I:%M:%S')
f.setLevel(logging.DEBUG)
f.setFormatter(formatter)

c.setLevel(logging.INFO)
c.setFormatter(formatter)
logger.addHandler(c)
logger.addHandler(f)
logger.info("[START]")




boolmap = {"True": True, "False": False}
corpus_home = "corpus/"
entity_id_map = {}
for i, k in enumerate(readfile("data/el/embeddings/dict.entity")):
	entity_id_map[k.split("\t")[0].replace("ko.dbpedia.org/resource/", "")] = i
# entity_id_map = {k.split("\t")[0].replace("ko.dbpedia.org/resource/", ""): i for i, k in enumerate([x for x in readfile("data/el/embeddings/dict.entity")])}
id_entity_map = {v: k for k, v in entity_id_map.items()}