from ...utils.AbstractArgument import AbstractArgument
class EVArgs(AbstractArgument):
	def __init__(self):
		# KB
		self.kb = "data/el/kb_entities"
		# Embedding
		self.word_embedding_path = "data/embedding/wiki_stem"
		self.word_embedding_type = "glove"
		self.entity_embedding_path = "data/embedding/ent_1903"
		self.entity_embedding_type = "glove"
		self.char_embedding_dim = 50

		# Data load path
		# load is done before data initialization
		self.data_load_path = "data/ev/10000/10000"
		# self.data_load_path = None
		# Data
		# if data load path is defined, these things won't be evaluated
		self.data_path = "corpus/wiki_cwform_10000.json"
		self.fake_er_rate = 0.1
		self.fake_el_rate = 0.1
		self.fake_ec_rate = 0.1
		self.fake_cluster_rate = 0.4 # 40% of entity set will be fake entity cluster
		self.ctx_window_size = 5 # context window size of token
		self.filter_data_tokens = True

		self.max_jamo = 69 #

		# training batch size
		self.batch_size = 64

		self.force_pretrain = True
		self.pretrain_epoch = 20

		# Model specification
		# ER
		self.er_model = "LSTM"
		self.er_output_dim = 100

		# EL
		self.el_model = "LSTM"
		self.el_output_dim = 100

		# EC
		self.ec_model = None
		self.er_score_threshold = 0.5
		self.el_score_threshold = 0.5

		# transformer
		self.transformer = "avg"
		self.encode_sequence = True
		self.transform_dim = 200

		# Train config
		self.epoch = 20
		self.lr = 1e-4
		self.momentum = 0.9
		self.eval_per_epoch = 2

		self.chunk_size = 100


	@property
	def er_model_path(self):
		try:
			return "data/ev/"+self.model_name+"er_scorer_%s_%s_%d.pt" % (self.er_model, self.word_embedding_type, self.er_output_dim)
		except:
			return None

	@property
	def el_model_path(self):
		try:
			return "data/ev/"+self.model_name+"el_scorer_%s_%s_%d.pt" % (self.el_model, self.entity_embedding_type, self.el_output_dim)
		except:
			return None

	@property
	def ec_model_path(self):
		try:
			return "data/ev/"+self.model_name+"ec_scorer.pt"
		except:
			return None

	@property
	def joint_model_path(self):
		try:
			return "data/ev/"+self.model_name+"joint_scorer.pt"
		except:
			return None
	

	@property
	def transformer_model_path(self):
		try:
			return "data/ev/"+self.model_name+"transformer.pt"
		except:
			return None

	@property
	def validation_model_path(self):
		try:
			return "data/ev/"+self.model_name+"validation.pt"
		except:
			return None
