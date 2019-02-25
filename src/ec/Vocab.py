
class Vocabulary():
	def __init__(self, surface, context_entities):
		self.surface = surface
		self.context = context_entities

	def get_embedding(self):
		# word2vec embedding + entity in context embedding
