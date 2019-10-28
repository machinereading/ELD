from ...ds import Vocabulary
from ...utils import pickleload, readfile
from ..utils import ELDArgs

class DRBasedTypeGiver:
	def __init__(self, args: ELDArgs):
		self.type_restriction = []
		self.kb_types = pickleload(args.kb_type_file)
		self.possible_type_list = [x for x in readfile(args.kb_type_list)]
	def __call__(self, *tokens: Vocabulary):
		for token in tokens:
			possible_types = self.possible_type_list[:]
			for relation in token.relation:
				target_token = token.get_relative_token(relation.relative_index)
				target_token_types = self.kb_types[target_token.entity] if target_token.entity in self.kb_types else []
				for rule in self.type_restriction:
					possible_types = [x for x in possible_types if x not in rule]
			token.kb_type = possible_types

		return tokens

