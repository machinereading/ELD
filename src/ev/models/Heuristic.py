

"""
Heuristic-based functions
MUST FOLLOW I/O
Input:
	entity: set of entities that might indicate same entity
		each entity is a dict, which contains:
			surface: surface form of entity
			type: NER-output type
			score?
Output:
	tuple of (float, string), which are score ([0,1]), representative surface form
"""

# sample heuristic function
def typechecker(entities):
	for item in entities:
		if item["type"] == "PERSON":
			return (1, item["surface"])

class HeuristicBasedValidator():
	def __init__(self):
		self.heuristic_sequence = []

	def validate(self, entity_set):
		
		sv_list = [x(entity_set) for x in self.heuristic_sequence]
		return sv_list