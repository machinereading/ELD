from ..utils.KoreanUtil import stem_sentence

class Sentence():
	def __init__(self, sentence, tokenize_method=stem_sentence):
		self.original_sentence = sentence
		self.tokens = [Vocabulary(x) for x in tokenize_method(sentence)]
		for i, token in enumerate(self.tokens):
			token.lctx = self.tokens[:i]
			token.rctx = self.tokens[i+1:]

	def __str__(self):
		return " ".join([str(x) for x in self.tokens])

	
class Vocabulary():
	def __init__(self, surface):
		self.surface = surface
		self.is_entity = False
		self.entity = None
		self.lctx = []
		self.rctx = []

	def __str__(self):
		return self.surface

	@property
	def context_entities(self):
		lctx = [x for x in self.lctx if x.is_entity]
		rctx = [x for x in self.rctx if x.is_entity]
		return lctx, rctx
