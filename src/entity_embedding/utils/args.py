from ...utils import AbstractArgument

class EmbeddingArgs(AbstractArgument):
	def __init__(self):
		self.char_embedding_dim = 50
		self.word_embedding_type = None
		self.word_embedding_path = None
		self.entity_embedding_type = None
		self.entity_embedding_path = None
		self.lstm_layers = 2
		self.lstm_hidden_dim = 256
		self.lstm_dropout_rate = 0.4
		self.lstm_bidirectional = True
