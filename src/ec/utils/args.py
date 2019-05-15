from ...utils import AbstractArgument

class ECArgs(AbstractArgument):
	def __init__(self, model_name):
		self.model_name = model_name
		self.data_path = "corpus/coref_formatted/"
		self.attn_dim = 200
		self.embedding_path = "data/embedding/wiki_tok_glove_300"
		self.embedding_type = "glove"
		self.lstm_dim = 200
		self.hidden_dim = 150
		self.epoch = 100
		self.eval_epoch = 5
		self.batch_size = 64
		self.lr = 1e-4

	@property
	def model_path(self):
		return "models/ec/%s" % self.model_name
