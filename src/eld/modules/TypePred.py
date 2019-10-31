from ...ds import Vocabulary
from ...utils import readfile, jsonload
from ..utils import ELDArgs

from tqdm import tqdm

class TypeGiver:
	def __init__(self, kbt, t, d, r, top_filter=0):
		relation_prefix = "http://dbpedia.org/ontology/"
		self.kb_types = jsonload(kbt)
		self.possible_type_list = [x for x in readfile(t)]
		self.domain_restriction = {}
		self.filter_len = top_filter
		for item in d:
			for line in readfile(item):
				s, o = line.strip().split("\t")
				if o not in self.possible_type_list: continue
				s = s[len(relation_prefix):]
				if s not in self.domain_restriction:
					self.domain_restriction[s] = set([])
				self.domain_restriction[s].add(o)
		self.range_restriction = {}
		for item in r:
			for line in readfile(item):
				s, o = line.strip().split("\t")
				if o not in self.possible_type_list: continue
				s = s[len(relation_prefix):]
				if s not in self.range_restriction:
					self.range_restriction[s] = set([])
				self.range_restriction[s].add(o)

	def __call__(self, *tokens: Vocabulary):
		result = []
		for token in tokens:
			possible_types = {x: 1 for x in self.possible_type_list}
			for relation in token.relation:
				# target_token = token.get_relative_token(relation.relative_index)
				# print(token.entity, target_token.entity)
				# target_token_types = self.kb_types[target_token.entity] if target_token.entity in self.kb_types else []
				dr_restriction = self.domain_restriction if relation.outgoing else self.range_restriction
				# if len(target_token_types) == 0: continue
				if relation.relation not in dr_restriction: continue
				for item in dr_restriction[relation.relation]:
					possible_types[item] *= relation.score
			possible_types = sorted(list(filter(lambda x: x[1] < 1, possible_types.items())), key=lambda x:x[1], reverse=True)
			if self.filter_len > 0:
				possible_types = possible_types[:self.filter_len]
			result.append([x[0] for x in possible_types])
		return result

	def get_gold(self, *tokens: Vocabulary):
		return [self.kb_types[token.entity] if token.entity in self.kb_types else [] for token in tokens]

