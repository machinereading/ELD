class EVArgs():
	def __init__(self):
		# Embedding
		self.word_embedding_path = "data/embedding/wiki_raw_word"
		self.word_embedding_type = "glove"
		self.entity_embedding_path = "data/embedding/entity"
		self.entity_embedding_type = "glove"

		# Data load path
		# load is done before data initialization
		# self.data_load_path = "data/ev/10000/10000"
		self.data_load_path = None
		# Data
		# if data load path is defined, these things won't be evaluated
		self.data_path = "corpus/el_wiki/wiki_cwform_10000.json"
		self.fake_er_rate = 0.1
		self.fake_el_rate = 0.1
		self.fake_ec_rate = 0.1
		self.fake_cluster_rate = 0.4 # 40% of entity set will be fake entity cluster
		self.ctx_window = 5 # context window size of token

		# training batch size
		self.batch_size = 2

		self.pretrain_epoch = 20
		# Model load path
		# load is done before model initialization
		# changed to property
		# self.er_model_path = None
		# self.el_model_path = None
		# self.ec_model_path = None

		# Model specification
		# if model path is defined, these things won't be evaluated
		
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

		# Train config
		self.epoch = 10
		self.lr = 1e-4
		self.momentum = 0.9

	@classmethod
	def from_json(cls, json_file):
		from ...utils import jsonload
		args = EV_Args()
		if type(json_file) is str:
			json_file = jsonload(json_file)
		for attr, value in json_file:
			setattr(args, attr, value)
		return args

	@classmethod
	def from_config(cls, ini_file):
		import configparser
		c = configparser.ConfigParser()
		c.read(ini_file)
		args = EV_Args(verify=False)
		for attr, value in c.items():
			setattr(args, attr, value)
		return args


	def to_json(self):
		return self.__dict__

	@property
	def er_model_path(self):
		try:
			return "data/ev/"+self.model_name+"er_scorer.pt"
		except:
			return None

	@property
	def el_model_path(self):
		try:
			return "data/ev/"+self.model_name+"el_scorer.pt"
		except:
			return None

	@property
	def ec_model_path(self):
		try:
			return "data/ev/"+self.model_name+"ec_scorer.pt"
		except:
			return None

	@property
	def transformer_model_path(self):
		try:
			return "data/ev/"+self.model_name+"transformer.pt"
		except:
			return None

