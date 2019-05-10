from ...utils import AbstractArgument

class ELArgs(AbstractArgument):
	def __init__(self):
		self.mode = "train"
		self.n_cands_before_rank = 30
		self.prerank_ctx_window = 50
		self.keep_p_e_m = 4
		self.keep_ctx_ent = 4
		self.ctx_window = 100
		self.tok_top_n = 25
		self.mulrel_type = "ment-norm"
		self.n_rels = 5
		self.hid_dims = 100
		self.snd_local_ctx_window = 6
		self.dropout_rate = 0.3
		self.n_epochs = 100
		self.dev_f1_change_lr = 0.68
		self.n_not_inc = 10
		self.eval_after_n_epochs = 5
		self.learning_rate = 1e-4
		self.margin = 0.01
		self.df = 0.5
		self.n_loops = 10
		self.train_filter_rate = 0.0

		# data
		self.ent_list_path = "data/el/kb_entities"
		self.entity_dict_path = "data/el/wiki_entity_dict.pickle"
		self.redirects_path = "data/el/redirects.pickle"
		self.word_voca_path = 'data/el/embeddings/dict.word'
		self.word_embedding_path = 'data/el/embeddings/word_embeddings.npy'
		self.snd_word_voca_path = 'data/el/embeddings/glove/dict_no_pos.word'
		self.snd_word_embedding_path = 'data/el/embeddings/glove/word_embeddings.npy'

		self.entity_voca_path = 'data/el/embeddings/dict.entity'
		self.entity_embedding_path = 'data/el/embeddings/entity_embeddings.npy'

	@property
	def model_path(self):
		return "models/el/" + self.model_name
