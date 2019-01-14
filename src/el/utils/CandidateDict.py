import json
import math
class CandidateDict():
	def __init__(self):
		self.surface_dict = {}
		self.entity_set = set([])
		self.link_modifier = 0.2

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
				print("Not a valid query!")
				return None
		elem = []
		if query in self.surface_dict:
			e = self.surface_dict[query]
			if e.exact_entity is not None:
				elem += [(e.exact_entity, 1)]
			elem += [(ent, score * self.link_modifier) for ent, score in self.surface_dict[query]]
		
		if len(elem) == 0:
			# more specified search
			# exact match could be skipped since it is already constructed
			# containing search & sharing word search
			words = [word for word in re.sub("[()_,.<>/?!\-]", " ", query).split(" ") if len(word) > 0]
			for entity in self.entity_dict:
				# length 1, 2 words are worthless to get containing entities
				if len(query) > 2 and query in entity or entity in query:
					elem.append((entity, 0.3))
				ent_words = [word for word in re.sub("[()_,.<>/?!\-]", " ", entity).split(" ") if len(word) > 0]
				containing_score = 0
				for word in words:
					if word in ent_words:
						containing_score += 1
				if containing_score > 0:
					elem.append((entity, containing_score / len(words)))
		if len(elem) > 0:
			elem = self.normalized_candidates(elem)

		return elem

	def __contains__(self, query):
		if type(query) is not str:
			try:
				query = str(query)
			except:
				return False
		return query in self.surface_dict

	def __len__(self):
		return len(self.surface_dict)

	def to_dict(self):
		return [v.to_dict() for v in self.surface_dict.values()]

	@classmethod
	def from_file(cls, f):
		result = CandidateDict()
		for v in json.load(f):
			result.add_candidate(LinkElem.from_dict(v))
		return result

	@classmethod
	def load_entity_from_file(cls, f, d=None):
		result = CandidateDict() if d is not None else d
		for line in f.readlines():
			result.entity_set.add(line.strip())
		return result

	def items(self):
		for k, v in self.surface_dict.items():
			yield k, v

	def normalized_candidates(self, candidates):
		score_sum = sum(map(lambda x: math.exp(x[1]), candidates))
		return [(x, math.exp(y) / score_sum * 0.95) for x, y in candidates] + [("#UNK#", 0.05)]



class LinkElem():
	def __init__(self, surface, exact_entity=None):
		self.surface = surface
		self.exact_entity = exact_entity
		self.link_dict = {}
		

	def add_link(self, entity, weight=1):
		if entity not in self.link_dict:
			self.link_dict[entity] = 0
		self.link_dict[entity] += weight

	def __iadd__(self, other):
		if type(other) is not LinkElem:
			return self
		if other.surface != self.surface:
			return self
		for k, w in other.link_dict.items():
			self.add_link(k, w)
		return self

	def __iter__(self):
		# for usage like "for item in candidate["Apple_Inc"]:"
		# yield tuples of (entity, score)
		# softmax is not applied here
		s = sum(self.link_dict.values())

		for item in  [(k, v / s) for k, v in self.link_dict]:
			yield item

	def postprocess(self, entity_list, redirect_list, disambiguation_list):
		# filter non-kb items and construct containing entities
		queue = list(self.link_dict.keys())[:]
		if self.exact_entity is not None and self.exact_entity not in entity_list:
			self.exact_entity = None
		elif self.surface.replace(" ", "_") in entity_list:
			self.exact_entity = self.surface.replace(" ", "_")

		# for item in entity_list:
		# 	words = item.split("_")

		r = redirect_list.keys()
		d = disambiguation_list.keys()
		i = 0
		end_list = set([])
		while len(queue) > 0:
			# print(queue)
			k = queue[0]
			del queue[0]
			if k not in self.link_dict:
				continue
			if k in r:
				t = redirect_list[k]
				v = self.link_dict[k]
				del self.link_dict[k]
				self.link_dict[t] = v
				if t not in end_list:
					queue.append(t)
					end_list.add(t)
			elif k in d:
				t = disambiguation_list[k]
				v = self.link_dict[k]
				del self.link_dict[k]
				for ii in t:
					self.link_dict[ii] = v / len(t)
					if ii not in end_list:
						queue.append(ii)
						end_list.add(ii)
			elif k not in entity_list:
				del self.link_dict[k]

		# not_present_keys = list(self.link_dict.keys())
		# if self.exact_entity is not None and self.exact_entity in not_present_keys:
		# 	not_present_keys.remove(self.exact_entity)
		# for entity in entity_list:
		# 	if entity in not_present_keys:
		# 		not_present_keys.remove(entity)
		# 	# if self.surface in entity:
		# 	# 	self.containing_entity.add(entity)
		# self.link_dict = {k: v for k, v in self.link_dict.items() if k not in not_present_keys}

	def __hash__(self):
		return hash(self.surface)

	def to_dict(self):
		return {
			"surface": self.surface,
			"link_dict": self.link_dict,
			"exact_entity": self.exact_entity
		}

	@classmethod
	def from_dict(cls, d):
		result = LinkElem(d["surface"])
		result.link_dict = d["link_dict"]
		result.exact_entity = d["exact_entity"]
		# result.containing_entity = set(d["containing_entity"])
		return result