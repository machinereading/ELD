class Vocabulary():
	def __init__(self, surface, parent_sentence, token_ind=0, char_ind=0):
		self.surface = surface
		self.is_entity = False
		self.entity = None
		self.entity_in_kb = False
		self.char_ind = char_ind
		self.token_ind = token_ind
		self.parent_sentence = parent_sentence

		# reserved for entity validation
		self.cluster = None
		self.error_type = -1 # -1: normal, 0: ER, 1: EL, 2: EC

	def __str__(self):
		return self.surface

	def __len__(self):
		return len(self.surface)

	def __repr__(self):
		return "%s: %s" % (self.surface, self.entity if self.entity is not None else "")

	@property
	def lctx_ent(self):
		return [x for x in self.lctx if x.is_entity]

	@property
	def rctx_ent(self):
		return [x for x in self.rctx if x.is_entity]

	@property
	def lctx(self):
		return self.parent_sentence.tokens[:self.token_ind]

	@property
	def rctx(self):
		return self.parent_sentence.tokens[self.token_ind+1:]

	def to_json(self):
		return {
			"surface": self.surface,
			"entity": self.entity,
			"is_entity": self.is_entity,
			"entity_in_kb": self.entity_in_kb,
			"parent_sentence": self.parent_sentence.id,
			"cluster": self.cluster.id if self.cluster is not None else -1,
			"error_type": self.error_type,
			"token_ind": self.token_ind,
			"char_ind": self.char_ind
		}

	@classmethod
	def from_json(cls, json):
		voca = Vocabulary(json["surface"], None, json["token_ind"], json["char_ind"])
		for k, v in json.items():
			setattr(voca, k, v)
		return voca