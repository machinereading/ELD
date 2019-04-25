from .utils.args import IterationArgs
from ..utils import jsonload, TimeUtil
from .. import *
from ..el import EL
from ..ev import EV
import os
class IterationModule():
	# EV --> EL Rerun model
	def __init__(self, el_model_name, ev_model_name):
		self.args = IterationArgs()
		self.el_model = EL("test", el_model_name)
		self.ev_model = EV("test", ev_model_name)

	def run(self, corpus_dir):
		corpus = []
		for item in os.listdir(corpus_dir):
			corpus.append(jsonload(corpus_dir+item))
		corpus = Corpus.load_corpus(corpus, filter_nik=True)
		validated_corpus = self.ev_model(corpus)
		newents = []
		for cluster in list(filter(lambda x: x.kb_uploadable, validated_corpus.cluster_list)):
			newents.append(cluster.target_entity)
			for token in cluster:
				self.el_model.data.ent_dict.add_instance(token.surface, token.entity)
		# remake entity embedding
		# TODO
		ent_embedding = None
		pass

		# update entity embedding
		self.el_model.data.update_ent_embedding(newents, ent_embedding)

		# reload EL
		self.el_model.reload_ranker()
		# retrain ???
		self.el_model
		# retrain corpus ???

		# run EL again
		el_result = self.el_model(corpus)

		# evaluate



	def _run_ev(self, corpus):
		return self.ev_model(corpus)

	def _update(self):
		# update el kb with validated items
		# update entity embedding ?
		# retrain EL ?
		# test EL
		pass

	def save(self):
		# save run result = updated kb
		pass