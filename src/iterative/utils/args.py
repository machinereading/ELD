from ...utils import AbstractArgument
class IterationArgs(AbstractArgument):
	def __init__(self):
		self.train_data_dir = ["corpus/crowdsourcing_processed/", "corpus/mta2_postprocessed/"]
		self.dev_data_dir = "corpus/el_golden_postprocessed_marked/"

	