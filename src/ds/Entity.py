
class Entity:
	def __init__(self, entity_form):
		self.entity_form = entity_form
		self.surface_form = set([])
		self.relation = []
		self._tensor = None

	@property
	def tensor(self):
		if self._tensor is not None:
			return self._tensor
		else:
			... # TODO calculate tensor

