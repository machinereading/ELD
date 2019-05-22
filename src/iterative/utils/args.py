from ...utils import AbstractArgument
import os
class IterationArgs(AbstractArgument):
	def __init__(self):
		self.ev_model_name = ""
		self.el_model_name = ""
		self.train_data_dir = ["corpus/crowdsourcing_processed/", "corpus/mta2_postprocessed/"]
		self.dev_data_dir = "corpus/el_golden_postprocessed_marked/"


		self.validation_data = ["corpus/crowdsourcing_1903_formatted/"+f for f in os.listdir("corpus/crowdsourcing_1903_formatted/") if "_dev" not in f]
		self.test_data = "corpus/iter_el_gold_dark_marked.json"
		self.word_embedding_path = "data/embedding/wiki_stem"
		self.word_embedding_type = "glove"
		self.entity_embedding_path = "data/embedding/entity_kbox_gl"
		self.entity_embedding_type = "glove"
