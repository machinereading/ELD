import os

from .utils import IterationArgs, EmbeddingCalculator
from .. import GlobalValues as gl
from ..ds import *
from ..ec import EC
from ..el import EL
from ..ev import EV
from ..utils import jsonload

class IterationModule:
	# EV --> EL Rerun model
	def __init__(self, el_model_name, ec_model_name, ev_model_name):
		gl.logger.info("Initializing IterationModule")
		self.args = IterationArgs()

		c = 0
		self.el_model_name = el_model_name
		self.el_model = EL("test", el_model_name)
		self.ec_model = EC("test", ec_model_name)
		self.ev_model = EV("test", ev_model_name)
		self.ent_cal = EmbeddingCalculator(self.ev_model.word_embedding, self.ev_model.entity_embedding)

	def run(self):
		gl.logger.info("Running IterationModule")
		gl.logger.debug("Loading corpus")
		train_corpus = [jsonload(self.args.train_data_dir + item) for item in os.listdir(self.args.train_data_dir)]
		validation_corpus = [jsonload(self.args.dev_data_dir + item) for item in os.listdir(self.args.dev_data_dir)]
		test_corpus = [jsonload(self.args.test_data + item) for item in os.listdir(self.args.test_data)]
		corpus = Corpus.load_corpus(validation_corpus)

		gl.logger.debug("Linking corpus")
		linked = self.el_model(corpus)
		gl.logger.debug("Clustering corpus")
		clustered = self.ec_model(linked)
		gl.logger.debug("Validating corpus")
		validated = self.ev_model(clustered)

		gl.logger.debug("Generating new embedding")
		new_cluster = list(filter(lambda x: x.kb_uploadable, validated.cluster_list))
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
		self.el_model.model_name = self.el_model_name + "_iter"
		self.el_model.args.model_name = self.el_model_name + "_iter"
		self.el_model.reload_ranker()

		# retrain ???
		gl.logger.info("Retraining EL Module")
		self.el_model.train(train_corpus, validation_corpus)

		# retrain corpus ???

		# run EL again
		el_result = self.el_model(test_corpus)

		# evaluate
		for sentence in el_result:
			for token in sentence:
				pass
