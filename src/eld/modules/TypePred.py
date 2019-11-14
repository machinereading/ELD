from ...ds import Vocabulary
from ...utils import readfile, jsonload

class TypeGiver:
	def __init__(self, kbt, t, d=(), r=(), dr=(), top_filter=0, use_dr=False, use_ne=True, use_hierarchy=True, mode="intersect"):
		assert mode in ["intersect", "union"]
		self.relation_prefix = "http://dbpedia.org/ontology/"
		self.kb_types = jsonload(kbt)
		self.possible_type_list = [x for x in readfile(t)]
		self.domain_restriction = {}
		self.range_restriction = {}
		self.dr_restriction = {}
		self.filter_len = top_filter

		self.hierarchical_types = jsonload("data/eld/typerefer/dbo_class_info.json")
		self.ne_tag_mapping = jsonload("data/eld/typerefer/ne_tag_info.json")
		self.use_dr = use_dr
		self.use_ne = use_ne
		self.use_hierarchy = use_hierarchy
		self.mode = mode
		for item in d:
			try:
				for line in readfile(item):
					s, o = line.strip().split("\t")
					if o not in self.possible_type_list: continue
					s = s[len(self.relation_prefix):]
					if s not in self.domain_restriction:
						self.domain_restriction[s] = set([])
					self.domain_restriction[s].add(o)
			except:
				pass
		for item in r:
			try:
				for line in readfile(item):
					s, o = line.strip().split("\t")
					if o not in self.possible_type_list: continue
					s = s[len(self.relation_prefix):]
					if s not in self.range_restriction:
						self.range_restriction[s] = set([])
					self.range_restriction[s].add(o)
			except:
				pass
		for item in dr:
			try:
				for line in readfile(item):
					rel, d, r = line.strip().split("\t")
					if d not in self.possible_type_list: continue
					if r not in self.possible_type_list: continue
					rel = rel[len(self.relation_prefix):]
					if d not in self.range_restriction:
						self.range_restriction[rel] = set([])
					self.range_restriction[rel].add((d, r))
			except:
				import traceback
				traceback.print_exc()

	def __call__(self, *tokens: Vocabulary):
		result = []

		for token in tokens:
			types = set([])
			if self.use_dr:
				if self.mode == "intersect":
					possible_types = set(self.possible_type_list)
				elif self.mode == "union":
					possible_types = set([])
				else:
					raise NotImplementedError
				for relation in token.relation:

					dr_restriction = self.domain_restriction if relation.outgoing else self.range_restriction
					# pair = relation.o if relation.outgoing else relation.s

					# if len(target_token_types) == 0: continue
					if relation.relation not in dr_restriction and relation.relation not in self.dr_restriction: continue
					if self.mode == "intersect":
						if len(dr_restriction[relation.relation]) > 0:
							possible_types &= set(dr_restriction[relation.relation])
					elif self.mode == "union":
						possible_types |= dr_restriction[relation.relation]
					# if relation.relation in self.dr_restriction and len(self.dr_restriction[relation.relation]) > 0:
					# 	if pair.entity not in self.kb_types: continue
					# 	pair_type = self.kb_types[pair.entity]
					# 	pair_idx = 1 if relation.outgoing else 0
					# 	my_idx = 0 if relation.outgoing else 1
					# 	for restriction in self.dr_restriction[relation.relation]:
					# 		if restriction[pair_idx] == pair_type:
					# 			possible_types[restriction[my_idx]] = 1 if self.mode == "intersect" else 0.999

				# possible_types = sorted(list(filter(filter_condition, possible_types.items())), key=lambda x:x[1], reverse=True)
				# if self.filter_len > 0:
				# 	possible_types = possible_types[:self.filter_len]
				# possible_types = set([x[0] for x in possible_types])
				possible_types = set(filter(lambda x: x in self.possible_type_list, possible_types))
				if len(possible_types) == len(self.possible_type_list):
					possible_types = set([])
				types |= possible_types
			if self.use_ne:
				if token.ne_type in self.ne_tag_mapping:
					types |= set(map(lambda x: "http://dbpedia.org/ontology/" + x, self.ne_tag_mapping[token.ne_type]["mapped_dbo"]))
					types.add("http://dbpedia.org/ontology/" + self.ne_tag_mapping[token.ne_type]["type"])
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

