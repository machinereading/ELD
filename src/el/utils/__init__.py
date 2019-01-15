from .CandidateDict import CandidateDict

with open("data/el/wiki_entity_dict.json", encoding="UTF8") as f:
	candidate_dict = CandidateDict.from_file(f)
with open("data/el/kb_entities", encoding="UTF8") as f:
	candidate_dict = CandidateDict.load_entity_from_file(f, candidate_dict)