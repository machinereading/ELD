from typing import List
from ...ds import Graph
from ...utils import readfile
class RelationCandDict:
	"""
	Generate candidate from neighbor entities
	"""
	def __init__(self, relation_file, entity_file=None):
		self.kg = Graph()
		if entity_file is not None:
			if type(entity_file) is str: entity_file = readfile(entity_file)
			for entity in entity_file:
				self.kg.add_node(entity)
		self.kg.add_kb_file(relation_file, add_new_node=entity_file is not None)

	def __getitem__(self, neighbor_entities: List):
		cands = {}
		for entity in self.kg.nodes.values():
			# cands[entity.name] = len([x for x in entity.incoming_edges.values() if x.e1 == ])
			pass
		return sorted([[k, v] for k, v in cands.items()], key=lambda x: -x[1])[:10]
