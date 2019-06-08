from ..el import EL
from ..ec import EC
from ..ev import EV
from ..ds import Corpus, Sentence
from .. import GlobalValues as gl
from ..utils import TimeUtil
from .postprocess import postprocess
from flask import Flask, request
import json
import jpype

gl.logger.info("Initializing demo")

app = Flask(__name__)

ELModule = EL("demo", "glove_entity_emb")
ECModule = EC("demo")
EVModule = EV("demo", "model_nn_explicit")
jpype.startJVM(jpype.getDefaultJVMPath())
@app.route("/eld/", methods=["POST"])
def eld():
	jpype.attachThreadToJVM()
	text = request.json
	return run(text)

def run(text):
	out_text = []
	if len(text) == 1 and "content" in text: # plain text
		text = text["content"]
		out_text.append(text)
	else:
		for sent in text:
			out_text.append(sent["text"])
	gl.logger.info("Input: %s" % text)
	gl.logger.debug("Running EL")
	with TimeUtil.TimeChecker("API"):
		el = ELModule(text)
		print(el)
		corpus = Corpus()
		for item in el:
			corpus.add_sentence(Sentence.from_cw_form(item))
		print(corpus.sentences[0].to_json())
		gl.logger.debug("Running EC")
		corpus = ECModule(corpus)
		gl.logger.debug("Running EV")
		corpus = EVModule(corpus)
		gl.logger.debug("Generating output")
		for cluster in list(filter(lambda x: x.kb_uploadable, corpus.cluster_list)):
			surface_count = {}
			max_surface = None
			max_surface_count = 0
			for elem in cluster:
				if elem.surface not in surface_count:
					surface_count[elem.surface] = 0
				surface_count[elem.surface] += 1
				if surface_count[elem.surface] > max_surface_count:
					max_surface_count = surface_count[elem.surface]
					max_surface = elem.surface
			represent_form = max_surface
			for elem in cluster:
				for sentence in corpus.sentences:
					for entity in sentence:
						if elem.surface == entity.surface:
							entity.entity = "__" + represent_form
		result = []
		for o, sentence in zip(out_text, corpus.sentences):
			e = {
				"text"       : o,
				"entities"   : [],
				"dark_entity": [],
			}
			for item in sentence.entities:
				if item.entity.startswith("__"):
					e["dark_entity"].append(postprocess(item).demo_form)
				elif item.entity != "NOT_IN_CANDIDATE":
					e["entities"].append(postprocess(item).demo_form)
			result.append(e)
	return json.dumps(result, ensure_ascii=False)

