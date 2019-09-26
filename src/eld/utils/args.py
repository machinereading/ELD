from ...utils import AbstractArgument
class ELDArgs(AbstractArgument):
	def __init__(self, model_name: str):

		self.model_name = model_name
		self.device = "cuda"

		# training config
		self.train_corpus_dir = None
		self.dev_corpus_dir = None
		self.epochs = 100
		self.eval_per_epoch = 5

		# transformer config
		self.transformer_mode = "joint" # one of "joint" or "separate"

		self.use_character_embedding = True
		self.use_word_context_embedding = True
		self.use_entity_context_embedding = True
		self.use_relation_embedding = True
		self.use_type_embedding = True
		self.context_window_size = 5

		self.character_encoder = None
		self.word_encoder = None
		self.entity_encoder = None
		self.relation_encoder = None
		self.type_encoder = None

		# data path config
		self.word_file = None
		self.word_embedding_file = None
		self.entity_file = None
		self.entity_embedding_file = None
		self.relation_file = None
		self.relation_embedding_file = None
		self.type_file = None
		self.type_embedding_file = None
		self.ent_list_path = "data/el/kb_entities"
		self.entity_dict_path = "data/el/wiki_entity_dict.pickle"
		self.redirects_path = "data/el/redirects.pickle"

		# values that will be modified in runtime
		self.c_emb_dim = 0
		self.w_emb_dim = 0
		self.e_emb_dim = 0
		self.r_emb_dim = 0
		self.t_emb_dim = 0

		# encoding config
		self.c_enc_dim = 50
		self.w_enc_dim = 100
		self.e_enc_dim = 100
		self.r_enc_dim = 100
		self.t_enc_dim = 100


		# if entity has different embedding, modify entity embedding (to average)
		self.modify_entity_embedding = False
		self.modify_entity_embedding_weight = 0.1  # weight on new entity

		# evaluation config
		self.corpus_dir = None

		# run config
		self.use_relation_candidates = False

		# prediction config
		self.map_threshold = 0.5

	@property
	def model_path(self):
		return "models/eld/" + self.model_name
