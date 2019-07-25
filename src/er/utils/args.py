
class ERArgs:
	def __init__(self):
		self.mode = "LSTM"
		self.num_epochs = 100
		self.batch_size = 16
		self.hidden_size = 128
		self.tag_space = 128
		self.num_layers = 1
		self.num_filters = 30
		self.char_dim = 30
		self.learning_rate = 0.015
		self.decay_rate = 0.1
		self.gamma = 0.0
		self.p_rnn = [0.33, 0.5]
		self.p_in = 0.33
		self.p_out = 0.5
		self.bigram = True
		self.schedule = 1
		self.unk_replace = 0.0
		self.embedding = "glove"
		self.embedding_dict = "data/er/vectors.txt"
