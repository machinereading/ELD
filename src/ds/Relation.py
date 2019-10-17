class Relation:
	def __init__(self, relative_index: int, relation: str, score: float, outgoing: bool):
		self.relative_index = relative_index
		self.relation = relation
		self.score = score
		self.outgoing = outgoing

	@classmethod
	def from_cw_form(cls, rel):
		return cls(rel[0], rel[1], rel[2], True if rel[3] == "outgoing" else False)
