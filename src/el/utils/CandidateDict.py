import json

class CandidateDict():
	def __init__(self):
		self.surface_dict = {}

	def add_candidate(self, candidate_elem):
		surface = candidate_elem.surface
		if surface not in self.surface_dict:
			self.surface_dict[surface] = candidate_elem
		else:
			self.surface_dict[surface] += candidate_elem

	def __getitem__(self, query):
		if type(query) is not str:
			try:
				query = str(query)
			except:
				return None

		elem = self.surface_dict[query] if query in self.surface_dict else None

		if elem is None:
			# containing search TODO
			pass

	def __contains__(self, query):
		if type(query) is not str:
			try:
				query = str(query)
			except:
				return False
		return query in self.surface_dict

	def to_dict(self):
		return [v.to_dict() for v in self.surface_dict.values()]

	@classmethod
	def from_file(cls, f):
		result = CandidateDict()
		for v in json.load(f):
			result.add_candidate(CandidateElem.from_dict(v))



class CandidateElem():
	def __init__(self, surface, exact_entity=False):
		self.surface = surface
		self.exact_entity = None if not exact_entity else surface
		self.containing_entity = set([])
		self.link_dict = {}
		self.exact_entity_modifier = 0.4
		self.containing_entity_modifier = 0.3
		self.link_modifier = 0.2
		self.unk_modifier = 0.1
		

	def add_link(self, entity, weight=1):
		if entity not in self.link_dict:
			self.link_dict[entity] = 0
		self.link_dict[entity] += weight

	def __iadd__(self, other):
		if type(other) is not CandidateElem:
			return self
		if other.surface != self.surface:
			return self
		for entity in other.containing_entity:
			self.containing_entity.add(entity)
		for k, w in other.link_dict.items():
			self.add_link(k, w)
		return self

	def __iter__(self):
		# for usage like "for item in candidate["Apple_Inc"]:"
		# yield tuples of (entity, prob)
		pass

	def __hash__(self):
		return hash(self.surface)

	def to_dict(self):
		return {
			"surface": self.surface,
			"link_dict": self.link_dict,
			"exact_entity": self.exact_entity,
			"containing_entity": list(self.containing_entity)
		}

	@classmethod
	def from_dict(cls, d):
		result = CandidateElem(d["surface"])
		result.link_dict = d["link_dict"]
		result.exact_entity = d["exact_entity"]
		result.containing_entity = set(d["containing_entity"])
		return result
