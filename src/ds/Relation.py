class Relation:
	def __init__(self, owner, relative_index: int, relation: str, score: float, outgoing: bool):
		self.owner = owner
		self.relative_index = relative_index
		self.relation = relation
		self.score = score
		self.outgoing = outgoing

	@classmethod
	def from_cw_form(cls, tok, rel):
		return cls(tok, rel[0], rel[1], rel[2], True if rel[3] == "outgoing" else False)

	@property
	def s(self):
		try:
			if self.outgoing: return self.owner
			return self.owner.parent_sentence.entities[self.owner.entity_idx + self.relative_index]
		except:
			pass

	@property
	def p(self):
		return self.relation

	@property
	def o(self):
		if not self.outgoing: return self.owner
		return self.owner.parent_sentence.entities[self.owner.entity_idx + self.relative_index]
