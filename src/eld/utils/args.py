from ...utils import AbstractArgument

class ELDArgs(AbstractArgument):
	def __init__(self):

		# training config
		self.corpus_dir = None
		self.epochs = 100
		self.eval_per_epoch = 5


		# transformer config
		self.use_character_embedding = True
		self.use_word_context_embedding = True
		self.use_entity_context_embedding = True

