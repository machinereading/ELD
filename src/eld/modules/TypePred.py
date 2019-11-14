from ...ds import Vocabulary
from ...utils import readfile, jsonload

class TypeGiver:
	def __init__(self, kbt, t, d=(), r=(), top_filter=0, use_dr=False, use_ne=True, use_hierarchy=True, mode="intersect"):
		assert mode in ["intersect", "union"]
		self.relation_prefix = "http://dbpedia.org/ontology/"
		self.kb_types = jsonload(kbt)
		self.possible_type_list = [x for x in readfile(t)]
		self.domain_restriction = {}
		self.filter_len = top_filter

		self.hierarchical_types = jsonload("data/eld/typerefer/dbo_class_info.json")
		self.ne_tag_mapping = jsonload("data/eld/typerefer/ne_tag_info.json")
		self.use_dr = use_dr
		self.use_ne = use_ne
		self.use_hierarchy = use_hierarchy
		self.mode = mode
		for item in d:
			for line in readfile(item):
				s, o = line.strip().split("\t")
				if o not in self.possible_type_list: continue
				s = s[len(self.relation_prefix):]
				if s not in self.domain_restriction:
					self.domain_restriction[s] = set([])
				self.domain_restriction[s].add(o)
		self.range_restriction = {}
		for item in r:
			for line in readfile(item):
				s, o = line.strip().split("\t")
				if o not in self.possible_type_list: continue
				s = s[len(self.relation_prefix):]
				if s not in self.range_restriction:
					self.range_restriction[s] = set([])
				self.range_restriction[s].add(o)

	def __call__(self, *tokens: Vocabulary):
		result = []

		for token in tokens:
			types = set([])
			if self.use_dr:
				possible_types = {x: 1 for x in self.possible_type_list}
				for relation in token.relation:
					# target_token = token.get_relative_token(relation.relative_index)
					# print(token.entity, target_token.entity)
					# target_token_types = self.kb_types[target_token.entity] if target_token.entity in self.kb_types else []
					dr_restriction = self.domain_restriction if relation.outgoing else self.range_restriction
					# if len(target_token_types) == 0: continue
					if relation.relation not in dr_restriction: continue
					if self.mode == "intersect":
						if len(dr_restriction[relation.relation]) > 0:
							for k in possible_types.keys():
								if k not in dr_restriction[relation.relation]:
									possible_types[k] = 0
					else:
						for item in dr_restriction[relation.relation]:
							possible_types[item] *= relation.score
				if self.mode == "intersect":
					filter_condition = lambda x: x[1] > 0
				else:
					filter_condition = lambda x: x[1] < 1
				possible_types = sorted(list(filter(filter_condition, possible_types.items())), key=lambda x:x[1], reverse=True)
				if self.filter_len > 0:
					possible_types = possible_types[:self.filter_len]
				possible_types = set([x[0] for x in possible_types])
				types |= possible_types
			if self.use_ne:
				if token.ne_type in self.ne_tag_mapping:
					types |= set(self.ne_tag_mapping[token.ne_type]["mapped_dbo"])
					types.add(self.ne_tag_mapping[token.ne_type]["type"])
			if self.use_hierarchy:
				add_items = set([])
				for t in types:
					if t.startswith(self.relation_prefix):
						t = t[len(self.relation_prefix):]
						if t in self.hierarchical_types:
							for l in self.hierarchical_types[t]["full_label"].split("."):
								add_items.add(self.relation_prefix + l)
				types |= add_items

			result.append(list(types))

		return result

	def get_gold(self, *tokens: Vocabulary):
		return [self.kb_types[token.entity] if token.entity in self.kb_types else [] for token in tokens]

