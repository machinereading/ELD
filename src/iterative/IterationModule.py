from .. import GlobalValues as gl
from .utils import IterationArgs, EmbeddingCalculator
from ..utils import jsonload, TimeUtil
from ..ds import *
from ..el import EL
from ..ev import EV

import os
class IterationModule():
	# EV --> EL Rerun model
	def __init__(self, el_model_name, ev_model_name):
		gl.logger.info("Initializing IterationModule")
		self.args = IterationArgs()

		self.train_corpus = []
		self.dev_corpus = []
		c = 0
		for d in self.args.train_data_dir:
			for item in os.listdir(d):
				j = jsonload(d+item)
				if c % 10 != 0:
					self.train_codrpus.append(j)
				else:
					self.dev_corpus.append(j)
		self.el_model = EL("test", el_model_name)
		self.ev_model = EV("test", ev_model_name)
		self.ent_cal = EmbeddingCalculator(self.ev_model.word_embedding, self.ev_model.entity_embedding)
	
	def run(self, corpus_dir):
		gl.logger.info("Running IterationModule")
		gl.logger.debug("Loading corpus")
		corpus = []
		for item in os.listdir(corpus_dir):
			corpus.append(jsonload(corpus_dir+item))
		corpus = Corpus.load_corpus(corpus)
		
		gl.logger.debug("Validating corpus")
		validated_corpus = self.ev_model(corpus)
		
		gl.logger.debug("Generating new embedding")
		new_cluster = list(filter(lambda x: x.kb_uploadable, validated_corpus.cluster_list))
		new_ents = []
		new_embs = self.ent_cal.calculate_cluster_embedding(new_cluster)
		for cluster in new_cluster:
			new_ents.append(cluster.target_entity)
			for token in cluster:
				self.el_model.data.surface_ent_dict.add_instance(token.surface, token.entity)
		

		# update entity embedding
		gl.logger.debug("Updating entity embedding")
		self.el_model.data.update_ent_embedding(new_ents, new_embs)

		# reload EL
		gl.logger.debug("Reloading ranker")
		self.el_model.model_name = el_model_name+"_iter"
		self.el_model.args.model_name = el_model_name+"_iter"
		self.el_model.reload_ranker()

		# retrain ???
		gl.logger.info("Retraining EL Module")
		self.el_model.train(self.train_corpus, self.dev_corpus)
		
		# retrain corpus ???

		# run EL again
		el_result = self.el_model(corpus)

		# evaluate
		for item in el_result:
			pass


	def _run_ev(self, corpus):
		return self.ev_model(corpus)
