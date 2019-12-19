import json
from flask import Flask, request
import jpype

from src.ds import Vocabulary, Relation
from src.eld.modules.TypePred import TypeGiver

app = Flask(__name__)


jpype.startJVM(jpype.getDefaultJVMPath())
DATA_PATH = ""
kbt = DATA_PATH + "entity_types.json"
t = DATA_PATH + "dbpedia_types"
d = DATA_PATH + "inst-d.tsv"
r = DATA_PATH + "inst-r.tsv"
typegiver = TypeGiver(kbt, t, [d], [r], use_dr=True, use_ne=True, use_hierarchy=True, mode="intersect")
@app.route("/typepred/", methods=["POST"])
def eld():
	jpype.attachThreadToJVM()
	text = request.json
	return run(text)

def run(j):
	vocas = {}
	for triple in j["PL"]["triples"]:
		s = triple["s"]
		p = triple["p"]
		o = triple["o"]
		if s not in vocas:
			vocas[s] = Vocabulary("", None, 0, 0)
		if o not in vocas:
			vocas[o] = Vocabulary("", None, 0, 0)
		s_voca = vocas[s]
		o_voca = vocas[o]
		s_voca.relation.append(Relation(s_voca, 0, p, 1, True))
		o_voca.relation.append(Relation(o_voca, 0, p, 1, False))
	rels = {k: typegiver(v) for k, v in vocas.items()}
	return json.dumps(rels, ensure_ascii=False)

