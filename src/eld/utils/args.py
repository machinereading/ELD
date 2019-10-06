from ...utils import AbstractArgument

class ELDArgs(AbstractArgument):
	def __init__(self, model_name: str = ""):
		self.model_name = model_name
		self.device = "cuda"

		# training config
		self.train_corpus_dir = "corpus/namu_eld_handtag_train/"
		self.dev_corpus_dir = "corpus/namu_eld_handtag_dev/"
		self.test_corpus_dir = "corpus/namu_eld_handtag_test/"
		self.epochs = 100
		self.eval_per_epoch = 5

		# transformer config
		self.transformer_mode = "separate"  # one of "joint" or "separate"

		self.use_character_embedding = True
		self.use_word_embedding = True
		self.use_word_context_embedding = True
		self.use_entity_context_embedding = True
		self.use_relation_embedding = True
		self.use_type_embedding = True
		self.context_window_size = 5

		self.character_encoder = None
		self.word_encoder = None
		self.word_context_encoder = None
		self.entity_context_encoder = None
		self.relation_encoder = None
		self.type_encoder = None

		# data path config
		self.character_file = "data/eld/char"
		self.character_embedding_file = "data/eld/character_embedding.npy"
		self.word_file = "data/embedding/wiki_stem.word"
		self.word_embedding_file = "data/embedding/wiki_stem.npy"
		self.entity_file = "data/eld/entities"
		self.entity_embedding_file = "data/eld/entity_embeddings.npy"
		self.relation_file = "data/eld/relations"  # 그냥 정의문
		self.type_file = "data/eld/types"  # 정의문
		self.ent_list_path = "data/eld/entities"
		self.entity_dict_path = "data/el/wiki_entity_dict.pickle"
		self.redirects_path = "data/el/redirects.pickle"
		# embedding config
		self.relation_limit = 5

		# encoding config
		self.c_enc_dim = 50
		self.w_enc_dim = 100
		self.e_enc_dim = 100
		self.r_enc_dim = 100
		self.t_enc_dim = 100

		# values that will be modified in runtime
		self.c_emb_dim = 50
		self.w_emb_dim = 0
		self.e_emb_dim = 0
		self.r_emb_dim = 0
		self.t_emb_dim = 0

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
