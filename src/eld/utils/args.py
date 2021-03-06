from ...utils import AbstractArgument

class ELDArgs(AbstractArgument):
	def __init__(self, model_name: str = ""):
		self.model_name = model_name
		self.device = "cuda"

		# training config
		# self.corpus_dir = "corpus/namu_eld_inputs_handtag_only/"
		# self.train_filter = "corpus/namu_handtag_only_train"
		# self.dev_filter = "corpus/namu_handtag_only_dev"
		# self.test_filter = "corpus/namu_handtag_only_test"
		self.epochs = 300
		self.eval_per_epoch = 2
		self.early_stop = 30
		self.train_corpus_limit = None
		self.dev_corpus_limit = None
		self.test_corpus_limit = None

		self.test_mode = False

		# transformer config
		# self.transformer_mode = "separate"  # one of "joint" or "separate"
		self.use_explicit_kb_classifier = True
		self.use_separate_feature_encoder = False
		self.use_surface_info = False
		self.use_candidate_info = False
		self.use_kb_relation_info = False
		self.use_cache_kb = True
		self.use_heuristic = True
		self.train_embedding = False
		self.use_character_embedding = True
		self.use_word_embedding = True
		self.use_word_context_embedding = True
		self.use_entity_context_embedding = True
		self.use_relation_embedding = True
		self.use_type_embedding = True
		self.context_window_size = 15

		self.character_encoder = "cnn"
		self.word_encoder = "cnn"
		self.word_context_encoder = "bilstm"
		self.entity_context_encoder = "bilstm"
		self.relation_encoder = "cnn"
		self.type_encoder = "ffnn"
		self.vector_transformer = "cnn"

		self.pred_train_mode = "mse"
		self.cand_only = False

		# data path config
		self.out_kb_entity_file = "data/eld/namu_no_ent_in_kb"
		self.character_file = "data/eld/char"
		self.character_embedding_file = "data/eld/character_embedding.npy"
		self.word_file = "data/eld/namu_tokenized.word"
		self.word_embedding_file = "data/eld/namu_tokenized.npy"
		self.entity_file = "data/eld/entities"
		# self.entity_file = "data/eld/handtag_only.word"
		self.entity_embedding_file = "data/eld/entity_embeddings.npy"
		# self.entity_embedding_file = "data/eld/handtag_only.npy"
		self.relation_file = "data/eld/relations"
		self.type_file = "data/eld/types"
		self.ent_list_path = "data/eld/entities"
		self.entity_dict_path = "data/el/wiki_entity_dict.pickle"
		self.redirects_path = "data/el/redirects.pickle"
		self.in_kb_linker = "pem"

		self.type_prediction = False
		self.kb_relation_file = "data/eld/kb_relations"
		self.kb_type_file = "data/eld/typerefer/entity_types.json"
		self.kb_type_list = "data/eld/typerefer/dbpedia_types"
		# self.domain_restriction_file = ["data/eld/typerefer/domain.tsv"]
		# self.range_restriction_file = ["data/eld/typerefer/range.tsv"]
		self.domain_restriction_file = ["data/eld/typerefer/inst-d.tsv"]
		self.range_restriction_file = ["data/eld/typerefer/inst-r.tsv"]
		# embedding config
		self.relation_limit = 5
		self.jamo_limit = 100
		self.word_limit = 5

		# encoding config
		self.c_enc_dim = 50
		self.w_enc_dim = 50
		self.wc_enc_dim = 100
		self.ec_enc_dim = 100
		self.r_enc_dim = 100
		self.t_enc_dim = 100

		# values that will be modified in runtime
		self.c_emb_dim = 0
		self.w_emb_dim = 0
		self.e_emb_dim = 0
		self.r_emb_dim = 0
		self.t_emb_dim = 0

		# if entity has different embedding, modify entity embedding (to average)
		self.modify_entity_embedding = False
		self.modify_entity_embedding_weight = 0.1  # weight on new entity

		# prediction config
		self.out_kb_threshold = 0.5
		self.new_ent_threshold = 0.3

	@property
	def model_path(self):
		return "models/eld/" + self.model_name

	@property
	def arg_path(self):
		return self.model_path + "_args.json"

	@property
	def flags(self):
		return len([x for x in [self.use_character_embedding, self.use_word_embedding, self.use_word_context_embedding, self.use_entity_context_embedding, self.use_relation_embedding, self.use_type_embedding] if x])
