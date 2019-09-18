from ...utils import AbstractArgument
from ... import GlobalValues as gl
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
		self.character_file = None
		self.character_embedding_file = None
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
		self.ce_dim = 0
		self.we_dim = 0
		self.ee_dim = 0
		self.re_dim = 0
		self.te_dim = 0

		# if entity has different embedding, modify entity embedding (to average)
		self.modify_entity_embedding = False
		self.modify_entity_embedding_weight = 0.1  # weight on new entity

		# evaluation config
		self.corpus_dir = None

		# run config
		self.use_relation_candidates = False

	@property
	def model_path(self):
		return "models/eld/" + self.model_name
