from ...utils import AbstractArgument
import os
class IterationArgs(AbstractArgument):
	def __init__(self):
		self.cluster_pickle = "data/iteration_el_ec.pkl"
		self.ev_model_name = ""
		self.el_model_name = ""
		self.train_data_dir = ["corpus/crowdsourcing_processed/", "corpus/mta2_postprocessed/"]
		self.dev_data_dir = "corpus/el_golden_postprocessed_marked/"

		self.validation_data_dir = "corpus/namu_gold_discovery/"
		self.test_data_dir = "corpus/namu_gold_test/"
		self.word_embedding_path = "data/embedding/wiki_stem"
		self.word_embedding_type = "glove"
		self.entity_embedding_path = "data/embedding/entity_kbox_gl"
		self.entity_embedding_type = "glove"

		self.put_fake_cluster = True
