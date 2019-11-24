import json
from flask import Flask, request
import jpype

from src.el import EL
from src.eld.DiscoveryOnly import DiscoveryModel
from src.eld.PredOnly import MSEEntEmbedding
from src.eld.modules.TypePred import TypeGiver
from src.ds import Corpus
from src.eld.utils import DataModule, ELDArgs

jpype.startJVM(jpype.getDefaultJVMPath())
app = Flask(__name__)
args = ELDArgs()
args.device = "cpu"
datamodule = DataModule("demo", args)
discovery = DiscoveryModel("demo", "discovery_degree_surface_4", data=datamodule)
el = EL(mode="demo")
ent_emb = MSEEntEmbedding("demo", "predtest", data=datamodule)
typepred = TypeGiver(kbt = "data/eld/typerefer/entity_types.json", t = "data/eld/typerefer/dbpedia_types", use_dr=False, use_ne=True, use_hierarchy=True)
print("Module loaded")
@app.route("/eld/", methods=["POST"])
def eld():
	jpype.attachThreadToJVM()
	text = request.json
	return run(text)

def generate_output(c: Corpus):
	sentence = c.sentences[0]

	result = {
		"text": sentence.original_sentence,
		"entities": [],
		"dark_entity": []
	}
	for entity in sentence.entities:
		ent = {
			"text": entity.surface,
			"start_offset": entity.char_idx,
			"end_offset": entity.char_idx + len(entity.surface),
			"ne_type": entity.ne_type,
			"type": entity.type_pred,
			"score": 0,
			"confidence": entity.confidence_score,
			"uri": entity.uri
		}
		if entity.is_dark_entity:
			result["entities"].append(ent)
		else:
			result["dark_entity"].append(ent)
	return result

def run(text):
	corpus = Corpus.from_string(text)
	discovery_result = discovery(corpus)

	return json.dumps(generate_output(corpus), ensure_ascii=False)

if __name__ == '__main__':
	print(run("《겨울왕국 2》(영어: Frozen 2)는 2019년 공개될 예정인 미국의 애니메이션 영화이다. 크리스 벅과 제니퍼 리가 감독하고 각본은 크리스벅, 제니퍼 리, 마크 E. 스미스, 크리스틴 앤더슨로페즈, 로버트 로페즈가 맡았다. 애니메이션 감독(Animation Supervisor)은 한국인 여성 이현민이 총괄한다. 2013년 영화 《겨울왕국》의 속편에 해당한다."))
