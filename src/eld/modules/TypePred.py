from ...ds import Vocabulary
from ...utils import pickleload, readfile, jsonload
from ..utils import ELDArgs

class TypeGiver:
	def __init__(self, args: ELDArgs):
		relation_prefix = "http://dbpedia.org/ontology/"
		self.kb_types = jsonload(args.kb_type_file)
		self.possible_type_list = [x for x in readfile(args.kb_type_list)]
		self.domain_restriction = {}
		for line in readfile(args.domain_restriction_file):
			s, o = line.strip().split("\t")
			if o not in self.possible_type_list: continue
			s = s[len(relation_prefix):]
			if s not in self.domain_restriction:
				self.domain_restriction[s] = set([])
			self.domain_restriction[s].add(o)

		self.range_restriction = {}
		for line in readfile(args.range_restriction_file):
			s, o = line.strip().split("\t")
			if o not in self.possible_type_list: continue
			s = s[len(relation_prefix):]
			if s not in self.range_restriction:
				self.range_restriction[s] = set([])
			self.range_restriction[s].add(o)

	def __call__(self, *tokens: Vocabulary):
		for token in tokens:
			possible_types = self.possible_type_list[:]
			target_filtered = False
			for relation in token.relation:
				target_token = token.get_relative_token(relation.relative_index)
				target_token_types = self.kb_types[target_token.entity] if target_token.entity in self.kb_types else []
				dr_restriction = self.domain_restriction if relation.outgoing else self.range_restriction
				if len(target_token_types) == 0: continue
				if relation not in dr_restriction: continue
				possible_types = list(filter(lambda x: x in dr_restriction[relation], possible_types))
				target_filtered = True
			if target_filtered:
				token.kb_type = possible_types
		return tokens

