from ..utils.KoreanUtil import stem_sentence
from .. import GlobalValues as gl
class Sentence():
	def __init__(self, sentence, tokenize_method=stem_sentence):
		self.original_sentence = sentence
		self.tokens = [Vocabulary(x, self, i) for i, x in enumerate(tokenize_method(sentence))]
		lastind = 0

		for i, token in enumerate(self.tokens):
			try:
				token.char_ind = self.original_sentence.index(token.surface, lastind)
				lastind = token.char_ind + len(token.surface)
			except:
				print(sentence, token)

	def __str__(self):
		return " ".join([str(x) for x in self.tokens])
	def __iter__(self):
		for token in self.tokens:
			yield token

	def add_ne(self, sin, ein, entity=None):
		def split_token(new_token, token):
			sin = new_token.char_ind
			ein = new_token.char_ind + len(new_token.surface)
			token_start = token.char_ind
			token_end = token.char_ind + len(token.surface)
			if sin <= token_start and token_end <= ein: return [new_token]
			if sin > token_end: return [token]
			if ein <= token.char_ind: return [token]
			# case 1: token start / sin / token end
			if token.char_ind < sin < token_end:
				tok1 = Vocabulary(token.surface[:sin-token.char_ind], self, char_ind=token.char_ind)
				return [tok1, new_token]
			# case 2: token start / ein / token end
			if token.char_ind < ein < token_end:
				tok1 = Vocabulary(token.surface[ein-token.char_ind:], self, char_ind=ein)
				return [new_token, tok1]
			# case 3: token start / sin / ein / token end
			tok1 = Vocabulary(token.surface[:sin-token.char_ind], self, char_ind=token.char_ind)
			tok2 = Vocabulary(token.surface[ein-token.char_ind:], self, char_ind=ein)
			return [tok1, new_token, tok2]

		new_token = Vocabulary(self.original_sentence[sin:ein], self, char_ind=sin)
		new_token.is_entity = True
		new_token.entity = entity
		new_token_list = []
		for token in self.tokens:
			new_token_list += split_token(new_token, token)
			if len(new_token_list) > 1 and new_token_list[-1] == new_token_list[-2]: new_token_list = new_token_list[:-1]
		for i, token in enumerate(new_token_list):
			token.token_ind = i
		self.tokens = new_token_list




	
class Vocabulary():
	def __init__(self, surface, parent_sentence, token_ind=0, char_ind=0):
		self.surface = surface
		self.is_entity = False
		self.entity = None
		self.char_ind = char_ind
		self.token_ind = token_ind
		self.parent_sentence=parent_sentence

	def __str__(self):
		return self.surface

	@property
	def context_entities(self):
		lctx = [x for x in self.lctx if x.is_entity]
		rctx = [x for x in self.rctx if x.is_entity]
		return lctx, rctx

	@property
	def lctx(self):
		return self.parent_sentence.tokens[:self.token_ind]

	@property
	def rctx(self):
		return self.parent_sentence.tokens[self.token_ind+1:]
