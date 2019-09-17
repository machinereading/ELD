from .Vocabulary import Vocabulary
from .. import GlobalValues as gl
from typing import Iterator
class Sentence:
	def __init__(self, sentence, tokenize_method=lambda x: x.split(" "), init=True):
		if not init: return
		self.original_sentence = sentence
		self.tokens = [Vocabulary(x, self, i) for i, x in enumerate(tokenize_method(sentence))]
		self.tagged_tokens = []

		self.id = -1
		lastind = 0
		self._vocab_tensors = None

		for i, token in enumerate(self.tokens):
			try:
				token.char_ind = self.original_sentence.index(token.surface, lastind)
				lastind = token.char_ind + len(token.surface)
			except:
				print(sentence, token)

	def __str__(self):
		return " ".join([str(x) for x in self.tokens])

	def __iter__(self) -> Iterator[Vocabulary]:
		for token in self.tokens:
			yield token

	def __len__(self):
		return len(self.tokens)

	def __getitem__(self, ind):
		return self.tagged_tokens[ind]

	@property
	def entities(self):
		return [x for x in self.tokens if x.is_entity]

	@property
	def not_in_kb_entities(self):
		return [x for x in self.tokens if x.is_entity and not x.entity_in_kb]

	@property
	def kb_entities(self):
		return [x for x in self.tokens if x.is_entity and x.entity_in_kb]

	def find_token_by_index(self, ind):
		for item in self.tokens:
			if item.char_ind == ind:
				return item
		return None

	def add_ne(self, sin, ein, surface, entity=None, cluster_id=-1):
		assert sin < ein
		assert self.original_sentence[sin:ein] == surface

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
				tok1 = Vocabulary(token.surface[:sin - token.char_ind], self, char_ind=token.char_ind)
				return [tok1, new_token]
			# case 2: token start / ein / token end
			if token.char_ind < ein < token_end:
				tok1 = Vocabulary(token.surface[ein - token.char_ind:], self, char_ind=ein)
				return [new_token, tok1]
			# case 3: token start / sin / ein / token end
			tok1 = Vocabulary(token.surface[:sin - token.char_ind], self, char_ind=token.char_ind)
			tok2 = Vocabulary(token.surface[ein - token.char_ind:], self, char_ind=ein)
			return [tok1, new_token, tok2]

		new_token = Vocabulary(self.original_sentence[sin:ein], self, char_ind=sin)
		new_token.ec_cluster_id = cluster_id
		new_token.is_entity = True
		new_token.entity = entity
		new_token.entity_in_kb = entity in gl.entity_id_map
		new_token_list = []
		for token in self.tokens:
			new_token_list += split_token(new_token, token)
			if len(new_token_list) > 1 and new_token_list[-1] == new_token_list[-2]: new_token_list = new_token_list[:-1]
		for i, token in enumerate(new_token_list):
			token.token_ind = i

		self.tokens = new_token_list

	def add_fake_entity(self, target_vocab):
		assert target_vocab in self.tokens

	@classmethod
	def from_cw_form(cls, cw_form):
		assert type(cw_form) is dict

		text = cw_form["text"]
		entities = cw_form["entities"]

		sentence = cls(text)
		if "fileName" in cw_form:
			sentence.id = cw_form["fileName"]
		error_count = 0
		# if len(entities) > 1000:
		# 	print(cw_form["fileName"])
		for entity in entities:
			if "entity" not in entity or entity["entity"] == "": continue
			try:
				cluster_id = entity["cluster"] if "cluster" in entity else -1
				sentence.add_ne(entity["start"], entity["end"], entity["surface"], entity["entity"], cluster_id)
			except:
				error_count += 1
		# if error_count > 0:
		# print(error_count, len(entities))
		# postprocess
		sentence.tokens = sorted(list(set(sentence.tokens)), key=lambda x: x.char_ind)
		return sentence

	def to_json(self):
		return {
			"original_sentence": self.original_sentence,
			"id"               : self.id,
			"tokens"           : [x.to_json() for x in self.tokens]
		}

	@classmethod
	def from_json(cls, json):
		sentence = cls(json["original_sentence"])
		sentence.id = json["id"]
		# if len(json["tokens"]) > 100000: print(json["original_sentence"], len(json["tokens"]))
		# print(len(json["tokens"]))
		sentence.tokens = [Vocabulary.from_json(x) for x in json["tokens"]]
		assert all([vocab.parent_sentence == sentence.id for vocab in sentence.tokens])
		for vocab in sentence.tokens:
			vocab.parent_sentence_id = sentence.id
			vocab.parent_sentence = sentence
		return sentence

	@property
	def vocab_tensors(self):
		if self._vocab_tensors is None:
			self._vocab_tensors = {
				"lctx_words"   : [],
				"rctx_words"   : [],
				"lctx_entities": [],
				"rctx_entities": [],
				"error_type"   : []
			}
			for vocab in self:
				self._vocab_tensors["lctx_words"].append(vocab.lctxw_ind)
				self._vocab_tensors["rctx_words"].append(vocab.rctxw_ind)
				self._vocab_tensors["lctx_entities"].append(vocab.lctxe_ind)
				self._vocab_tensors["rctx_entities"].append(vocab.rctxe_ind)
				self._vocab_tensors["error_type"].append(vocab.error_type)

		return self._vocab_tensors
